import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm._aiter_ops import rocm_aiter_ops
from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ParallelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.deep_gemm import fp8_mqa_logits, fp8_paged_mqa_logits
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager


if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._ipex_ops import ipex_ops as ops

import autort, os, random

def init_kda_fn(device, kda_path, buffer_count):
  import os
  from safetensors.torch import safe_open
  import vllm.distributed
  world_size = vllm.distributed.get_tp_group().world_size
  world_rank = vllm.distributed.get_tp_group().rank_in_group

  def from_float8_blockwise(w, ws, block_size=128, dtype=torch.bfloat16):
    shape = w.shape
    assert w.dtype == torch.float8_e4m3fn
    assert w.dim() == ws.dim() and w.dim() in (2, 3)
    if w.dim() == 2:
      w, ws = w.unsqueeze(0), ws.unsqueeze(0)
    else:
      assert w.size(0) == ws.size(0)
    ph = torch.empty([ws.size(0), ws.size(1) * block_size, ws.size(2) * block_size], dtype=w.dtype, device=w.device)
    ph[:, :w.size(1), :w.size(2)] = w
    ph = (ph.view(w.size(0), ws.size(1), block_size, ws.size(2), block_size).to(ws.dtype) * ws.view(w.size(0), ws.size(1), 1, ws.size(2), 1)).view(ph.shape)
    return ph[:, :w.size(1), :w.size(2)].to(dtype).view(shape)

  def load_tensor_fp8(f, key, device='cuda', fp8_to_bf16=False):
    w = f.get_tensor(key).to(device)
    try:
      w.scale_inv = f.get_tensor(key + '_scale_inv').float().to(device)
    except:
      pass
    if fp8_to_bf16 and w.dtype == torch.float8_e4m3fn:
      w = from_float8_blockwise(w, w.scale_inv)
    return w

  def world_slice(t, dim=0):
    if t is None or dim is None:
      return t
    assert t.size(dim) % world_size == 0, f'Failed during slicing tensor of shape {list(t.shape)} to {world_size} pieces at dim-{dim}.'
    group_size = t.size(dim) // world_size
    out = t.narrow(dim, world_rank * group_size, group_size).contiguous()
    if hasattr(t, 'scale_inv'):
      assert t.scale_inv.size(dim) % world_size == 0
      group_size = t.scale_inv.size(dim) // world_size
      out.scale_inv = t.scale_inv.narrow(dim, world_rank * group_size, group_size).contiguous()
    return out

  with safe_open(kda_path, framework='pt') as f:
    param = {}
    for k in f.keys():
      if '.kda.' not in k:
        continue
      if k.endswith('_scale_inv'):
        continue
      if '.kda.k_proj.' in k or '.kda.v_proj.' in k or '_a_proj.weight' in k:
        continue
      if '.kda.q_proj.' in k:
        k_new = k.replace('q_', 'fused_')
        param[k_new] = torch.cat([
          world_slice(load_tensor_fp8(f, k, 'cpu', fp8_to_bf16=True), dim=0),
          world_slice(load_tensor_fp8(f, k.replace('q_', 'k_'), 'cpu', fp8_to_bf16=True), dim=0),
          world_slice(load_tensor_fp8(f, k.replace('q_', 'v_'), 'cpu', fp8_to_bf16=True), dim=0),
          load_tensor_fp8(f, k.replace('q_', 'f_a_'), 'cpu', fp8_to_bf16=True),
          load_tensor_fp8(f, k.replace('q_', 'g_a_'), 'cpu', fp8_to_bf16=True),
        ]).to(device)
        continue
      if '.kda.o_proj.' in k:
        param[k] = world_slice(load_tensor_fp8(f, k, 'cpu', fp8_to_bf16=True), dim=1).to(device)
        continue
      if '.kda.o_norm.' in k:
        param[k] = load_tensor_fp8(f, k, 'cpu', fp8_to_bf16=True).to(device)
        continue
      param[k] = world_slice(load_tensor_fp8(f, k, 'cpu', fp8_to_bf16=True), dim=0).to(device)

    from fla2.modules import FusedRMSNormGated, ShortConvolution
    from fla2.ops.kda import fused_recurrent_kda
    from fla2.ops.kda.gate import fused_kda_gate
    n_layers, total_layers, num_heads, head_k_dim, head_v_dim, norm_eps = 6, 61, 128 // world_size, 256, 512, 1e-06
    key_dim, value_dim, conv_size = num_heads * head_k_dim, num_heads * head_v_dim, 4

    with torch.no_grad():
      kda_o_norm = [FusedRMSNormGated(head_v_dim, activation="sigmoid", eps=norm_eps).to(device) for i in range(n_layers)]
      for l in range(n_layers):
        kda_o_norm[l].weight.data.copy_(param.get(f'model.layers.{total_layers - n_layers + l}.kda.o_norm.weight', kda_o_norm[l].weight))

    '''
    q_conv1d = [ShortConvolution(hidden_size=key_dim, kernel_size=conv_size, bias=False, activation="silu").to(device) for i in range(n_layers)]
    k_conv1d = [ShortConvolution(hidden_size=key_dim, kernel_size=conv_size, bias=False, activation="silu").to(device) for i in range(n_layers)]
    v_conv1d = [ShortConvolution(hidden_size=value_dim, kernel_size=conv_size, bias=False, activation="silu").to(device) for i in range(n_layers)]
    for l in range(n_layers):
      q_conv1d[l].weight.data.copy_(param.get(f'model.layers.{l}.kda.q_conv1d.weight', q_conv1d[l].weight))
      k_conv1d[l].weight.data.copy_(param.get(f'model.layers.{l}.kda.k_conv1d.weight', k_conv1d[l].weight))
      v_conv1d[l].weight.data.copy_(param.get(f'model.layers.{l}.kda.v_conv1d.weight', v_conv1d[l].weight))
    conv1d_state = torch.empty([n_layers, max_concurrency, 2 + value_dim // key_dim, key_dim, conv_size], dtype=torch.bfloat16, device=device)

    if conv1d_state is not None:
      conv1d_state[:, v.constant].zero_()

    q = q_conv1d[l](hidden_states @ param[f'model.layers.{l}.kda.q_proj.weight'].t(),
      cache=conv1d_state[l, 0] if conv1d_state is not None else None, output_final_state=conv1d_state is not None)[0]
    k = k_conv1d[l](hidden_states @ param[f'model.layers.{l}.kda.k_proj.weight'].t(),
      cache=conv1d_state[l, 1] if conv1d_state is not None else None, output_final_state=conv1d_state is not None)[0]
    v = v_conv1d[l](hidden_states @ param[f'model.layers.{l}.kda.v_proj.weight'].t(),
      cache=conv1d_state[l, 2:].flatten(0, 1) if conv1d_state is not None else None, output_final_state=conv1d_state is not None)[0]
    '''
  recurrent_state = torch.zeros([n_layers, buffer_count, num_heads, head_k_dim, head_v_dim], dtype=torch.float32, device=device)
  return param, kda_o_norm, recurrent_state, n_layers, total_layers


@torch.compiler.disable(recursive=True)
def prepare_inflight_index_map(self, positions, kda_path, buffer_count=32, init_hook=None):
  forward_context = get_forward_context()
  if not hasattr(prepare_inflight_index_map, 'inflight_table_map'):
    device = positions.device
    os.environ['LOCAL_RANK'] = str(device.index)
    os.environ['RANK'] = str(torch.distributed.get_rank())
    os.environ['SIZE'] = str(torch.distributed.get_world_size())
    torch._dynamo.config.recompile_limit = 4096

    if init_hook is not None:
      init_hook(device)
    self.kda_tp_param, self.kda_o_norm, self.recurrent_state, self.n_layers, self.total_layers = init_kda_fn(device, kda_path, buffer_count)

    inflight_table_map = torch.full([192000], -1, dtype=torch.int32, device=device)
    inflight_table_map[0] = 0
    prepare_inflight_index_map.inflight_table_map = inflight_table_map
    prepare_inflight_index_map.inflight_map_fn = torch.compiler.disable(autort.export(name=f'inflight_map_fn', dev=device.index, source=r'''
@DEF_FUNC: query_start_loc:int32[N], positions:int64[P], block_table:int32[N, L], inflight_table_map:int32[NUMBLOCKS] -> current_map:int32[N]
@DEF_BIND: ~%~:1
@DEF_EXTRA: world_rank:int32, buffer_count:int32

void main() {
  for (int n = 0; n < size_of_N(); ++n) {
    int left = positions(query_start_loc(n)), right = positions(query_start_loc(n + 1) - 1) + 1;
    if ((left == 0 && right == 1) || block_table(n, 0) < 0) { // warmup
      current_map(n) = 0; continue;
    }
    // printf("debug[gpu-%d, sample-%d/%d]: positions=[%d .. %d)\n", world_rank, n, int(size_of_N()), left, right); continue;

#define TABLE_COUNTER()      (inflight_table_map(0))
#define LEADING_BLOCK_ID(n)  inflight_table_map(block_table(n, 0) + 1)

    if (left == 0) {
      // prefill update mapping
      int curr_addr = atomicAdd(&TABLE_COUNTER(), 1) % buffer_count;
      int prev_addr = LEADING_BLOCK_ID(n);
      LEADING_BLOCK_ID(n) = curr_addr;
      if (world_rank == 0)
        printf("[Inflight-Batch-Logging] New Request => [gpu-%d, sample-%d/%d]: forwarding positions=[%d .. %d), sample-%d register to new cache_idx-%d (previous-idx:%d)\n",
          world_rank, n, int(size_of_N()), left, right, n, curr_addr, prev_addr);
    } else {
      // non-leading-prefill or decode query mapping
      if (world_rank == 0) {
        if (left + 1 < right)
          printf("[Inflight-Batch-Logging] Old Request => [gpu-%d, sample-%d/%d]: forwarding positions=[%d .. %d), sample-%d should use cache_idx-%d\n",
            world_rank, n, int(size_of_N()), left, right, n, LEADING_BLOCK_ID(n));
        if (LEADING_BLOCK_ID(n) < 0)
          printf("  [error found] inflight batching has no mapping index.\n");
      }
    }
    current_map(n) = LEADING_BLOCK_ID(n);
  }
}'''))

  else:
    inflight_table_map = prepare_inflight_index_map.inflight_table_map

  if forward_context.attn_metadata is None:
    return None
  else:
    attn_metadata = forward_context.attn_metadata.get('model.layers.0.self_attn.indexer.k_cache', None)
    if attn_metadata is not None:
      query_start_loc = attn_metadata.query_start_loc[:-1]
      world_rank = int(torch.distributed.get_rank())

      map_array_1d = []
      req_offset = 0
      # NOTE: num_decodes IS BEFORE num_decodes in query_start_loc
      if attn_metadata.num_decodes > 0:
        req_offset += attn_metadata.num_decodes
        block_table = attn_metadata.decode.block_table
        assert block_table.size(0) == attn_metadata.num_decodes
        map_array_1d.append(prepare_inflight_index_map.inflight_map_fn(query_start_loc[:req_offset],
          positions, block_table, inflight_table_map, extra=[world_rank, buffer_count]))

      if attn_metadata.num_prefills > 0:
        for chunk in attn_metadata.prefill.chunks:
          block_table = chunk.block_table
          for i in range(block_table.size(0)):
            req_offset += 1
            map_array_1d.append(prepare_inflight_index_map.inflight_map_fn(query_start_loc[req_offset - 1:req_offset],
              positions, block_table[i:i + 1], inflight_table_map, extra=[world_rank, buffer_count]))

      assert attn_metadata.num_prefills + attn_metadata.num_decodes == req_offset
      self.kda_attn_metadata = attn_metadata
      self.kda_map_array_1d = map_array_1d
      return self
    else:
      return None

def paged_kda_forward(kda, x, metadata):
    map_tensor, attn_metadata = metadata
    assert x.dim() == 2
    o = torch.empty_like(x)
    if attn_metadata is not None:
        index_map_ptr = iter(map_tensor)
        if attn_metadata.num_decodes > 0:
            map_ptr = next(index_map_ptr)
            # cache_states = kda.cache_states.index_select(0, map_ptr)
            kda_output, _, _ = kda(
                  x[:map_ptr.size(0)].unsqueeze(1),
                  attention_mask=None,
                  past_key_value=None,
                  use_cache=False,
                  output_attentions=False,
            )
            o[:map_ptr.size(0)].copy_(kda_output.flatten(0, 1))
        if attn_metadata.num_prefills > 0:
            for chunk in attn_metadata.prefill.chunks:
              map_ptr = next(index_map_ptr)
              assert map_ptr.numel() == 1
              # cache_states = kda.cache_states.index_select(0, map_ptr)
              kda_output, _, _ = kda(
                  x[chunk.token_start:chunk.token_end].unsqueeze(0),
                  attention_mask=None,
                  past_key_value=None,
                  use_cache=False,
                  output_attentions=False,
              )
              o[chunk.token_start:chunk.token_end].copy_(kda_output.flatten(0, 1))
    else:
        o.fill_(-1)
    return o

def kda_layer(config, buffer_count=32):
    from fla2.layers.kda import KimiDeltaAttention
    return KimiDeltaAttention(
      hidden_size=getattr(config, 'hidden_size', 7168),
      num_heads=getattr(config, 'num_attention_heads', 128),
      head_dim=getattr(config, 'kda_head_dim', 256),
      expand_v=getattr(config, 'kda_expand_v', 2),
      use_short_conv=getattr(config, 'kda_use_short_conv', False),
      conv_size=getattr(config, 'kda_conv_size', 4),
      conv_bias=getattr(config, 'kda_conv_bias', False),
      norm_eps=getattr(config, 'rms_norm_eps', 1e-06),
      buffer_count=buffer_count
    )


def paged_kda_attn(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    layer_idx: int,
    state_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    # Safely call triton/fla/autort/custom ops within non-fake version, e.g.:
    assert hasattr(paged_kda_attn, 'kda_tp_param')
    if not hasattr(paged_kda_attn, 'kda_attn_metadata'):
      return hidden_states.clone()
    param, l = paged_kda_attn.kda_tp_param, layer_idx
    hidden_states_shape = hidden_states.shape

    kda_A_log = param.get(f'model.layers.{l}.kda.A_log', None)
    if kda_A_log is None:
      return hidden_states

    from fla2.modules import FusedRMSNormGated, ShortConvolution
    from fla2.ops.kda import fused_recurrent_kda
    from fla2.ops.kda.gate import fused_kda_gate
    world_size = get_tensor_model_parallel_world_size()
    num_heads, head_k_dim, head_v_dim, norm_eps = 128 // world_size, 256, 512, 1e-06
    key_dim, value_dim, conv_size = num_heads * head_k_dim, num_heads * head_v_dim, 4

    hidden_states = hidden_states.view(1, -1, hidden_states.size(-1))
    assert hidden_states.dim() == 3
    qkvfg = hidden_states @ param[f'model.layers.{l}.kda.fused_proj.weight'].t()
    qkv_dim = num_heads * (head_k_dim + head_k_dim + head_v_dim)
    fg_a_dim = (qkvfg.size(-1) - qkv_dim) // 2
    qkv = torch.nn.functional.silu(qkvfg.narrow(-1, 0, qkv_dim)).view(*qkvfg.shape[:-1], num_heads, -1) # bm,hkm->bhk

    beta = (hidden_states @ param[f'model.layers.{l}.kda.b_proj.weight'].t()) # .sigmoid()  # bs,hs->bh
    g = (qkvfg.narrow(-1, qkv_dim, fg_a_dim) @ param[f'model.layers.{l}.kda.f_b_proj.weight'].t()) # bs,lm->bl |  bl,hkl->bhk

    layer_offset = l - (paged_kda_attn.total_layers - paged_kda_attn.n_layers)
    state = paged_kda_attn.recurrent_state[layer_offset][:hidden_states.size(0)]
    g = fused_kda_gate(g=g.view(*qkvfg.shape[:-1], num_heads, -1), A_log=kda_A_log, dt_bias=param[f'model.layers.{l}.kda.dt_bias'])
    o, _ = fused_recurrent_kda(
                q=qkv.narrow(-1, 0, head_k_dim),
                k=qkv.narrow(-1, head_k_dim, head_k_dim),
                v=qkv.narrow(-1, head_k_dim * 2, head_v_dim),
                g=g,
                beta=beta,
                initial_state=state, # B,H,K,V
                use_qk_l2norm_in_kernel=True)

    f = torch.addmm(param[f'model.layers.{l}.kda.g_b_proj.bias'].view(1, -1), qkvfg.view(-1, qkvfg.size(-1))
            .narrow(-1, qkv_dim + fg_a_dim, fg_a_dim), param[f'model.layers.{l}.kda.g_b_proj.weight'].t())
    o = paged_kda_attn.kda_o_norm[layer_offset](o, f.view(*qkvfg.shape[:-1], num_heads, -1))
    o = o.flatten(-2) @ param[f'model.layers.{l}.kda.o_proj.weight'].t()  # bhk,mhk->bm
    return o.view(hidden_states_shape)


# using registered fake `torch.ops.vllm.custom_fn` to hide torch.compile.disable failure
def paged_kda_attn_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    layer_idx: int,
    state_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)

direct_register_custom_op(
    op_name="paged_kda_attn",
    op_func=paged_kda_attn,
    mutates_args=[],
    fake_impl=paged_kda_attn_fake,
    dispatch_key=current_platform.dispatch_key,
)

