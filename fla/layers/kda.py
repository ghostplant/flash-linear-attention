# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class KimiDeltaAttention(nn.Module):
    """
    Kimi Delta Attention (KDA) layer implementation.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dimension. Default: 1.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 128.
        num_heads (int, Optional):
            The number of heads. Default: 16.
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA (Grouped Value Attention) is applied if `num_v_heads` > `num_heads`. Default: `None`.
        mode (str, Optional):
            Which Kimi Delta Attention kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the beta will be multiplied by 2.
            See reference:
            [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int = None,
        mode: str = "chunk",
        use_short_conv: bool = False,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        buffer_count: int = 32,
    ) -> KimiDeltaAttention:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.zeros(self.key_dim, dtype=torch.float32))
        self.dt_bias._no_weight_decay = True
        self.cache_states = torch.zeros([buffer_count, self.num_heads, self.head_dim, self.head_v_dim], dtype=torch.float32)

        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        recurrent_state: torch.Tensor = None,
        mode: str = 'fused_recurrent',
        attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = True,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:

        batch_size, q_len, _ = hidden_states.shape
        q = F.silu(hidden_states @ self.q_proj.weight.t())  # bm,hkm->bhk
        k = F.silu(hidden_states @ self.k_proj.weight.t())  # bm,hkm->bhk
        v = F.silu(hidden_states @ self.v_proj.weight.t())  # bm,hkm->bhk

        g = hidden_states @ self.f_proj[0].weight.t() @ self.f_proj[1].weight.t()  # bs,lm->bl |  bl,hkl->bhk
        beta = hidden_states @ self.b_proj.weight.t()  # bs,hs->bh

        q, k, g = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k, g))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        if self.allow_neg_eigval:
            beta = beta.sigmoid() * 2.0
        else:
            beta = beta.sigmoid()

        if recurrent_state is None and self.cache_states.size(0) >= batch_size:
            recurrent_state = self.cache_states[:batch_size]

        if mode == "chunk":
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=self.A_log,     # h
                dt_bias=self.dt_bias, # hk
                initial_state=recurrent_state, # B,H,K,V
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
            )
        elif mode == "fused_recurrent":
            g = fused_kda_gate(g=g, A_log=self.A_log, dt_bias=self.dt_bias)
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state, # B,H,K,V
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        factor = hidden_states @ self.g_proj[0].weight.t() @ self.g_proj[1].weight.t() + self.g_proj[1].bias[None, None, :]  # bs,lm->bl | bl,hkl->bhk | bhk + hk->bhk
        factor = rearrange(factor, "... (h d) -> ... h d", d=self.head_v_dim)
        o = self.o_norm(o, factor)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = o @ self.o_proj.weight.t()  # bhk,mhk->bm

        return o, recurrent_state, _
