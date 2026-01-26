import torch
from fla.layers.kda import KimiDeltaAttention

torch.manual_seed(0)
config = {}

with torch.no_grad():
  kda = KimiDeltaAttention(
                    hidden_size=getattr(config, 'hidden_size', 7168),
                    num_heads=getattr(config, 'num_attention_heads', 128),
                    head_dim=getattr(config, 'kda_head_dim', 256),
                    expand_v=getattr(config, 'kda_expand_v', 2),
                    use_short_conv=getattr(config, 'kda_use_short_conv', False),
                    conv_size=getattr(config, 'kda_conv_size', 4),
                    conv_bias=getattr(config, 'kda_conv_bias', False),
                    norm_eps=getattr(config, 'rms_norm_eps', 1e-06),
                    layer_idx=60,
  ).bfloat16().cuda()
  x = torch.randn([8, 1, 7168]).bfloat16().cuda()
  state_cache = torch.zeros([31, kda.num_heads, kda.head_dim, kda.head_v_dim], dtype=torch.float32, device='cuda')
  print(kda(x, state_cache))
