import torch
import math
from typing import Tuple
from torch import nn
from src.core.layers.attention_pool import CausalSelfMHA
from src.core.layers.feedforward import gelu_ffn



class GPT2DecoderBlock(nn.Module):
    '''
    pre-layer_normalization 的 decoder block 架构图:
    
        -----------------add---------------->|       ------------add------------>|
      x --layer_norm-->|--causal_attention-->|--> x_ --layer_norm-->--gelu_ffn-->|--> y
    kv_cache(if any)-->|                      --> new_kv_cache(if need)
   attn_mask(if any)-->|
   positions(if any)-->|

    x / x_ / y have same shape
    '''
    def __init__(self,
                 embd_size:int,
                 num_heads:int,
                 use_bias:bool,
                 max_context_size:int,
                 attn_p_drop:float,
                 resid_p_drop:float,
                 use_rope:bool,
                 use_cached_causal_mask:bool):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embd_size, bias=use_bias)
        self.causal_attention = CausalSelfMHA(embd_size, num_heads, use_bias, max_context_size, attn_p_drop,
                                              resid_p_drop, use_rope, use_cached_causal_mask)
        self.layer_norm2 = nn.LayerNorm(embd_size, bias=use_bias)
        self.gelu_ffn = gelu_ffn(embd_size, use_bias, resid_p_drop)

    def forward(self,
                x:torch.Tensor,
                kv_cache:Tuple[torch.Tensor, torch.Tensor]|None = None,
                return_cache:bool = False,
                attention_mask:torch.Tensor|None = None,
                positions:torch.Tensor|None = None):
        
        attn_result, new_kv_cache = self.causal_attention(self.layer_norm1(x), kv_cache, return_cache, attention_mask, positions)
        x_ = x + attn_result
        y = x_ + self.gelu_ffn(self.layer_norm2(x_))
        
        return y, new_kv_cache