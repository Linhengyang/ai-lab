import torch
import math
from typing import Tuple
from torch import nn
from src.core.layers.attention_pool import MultiHeadAttention
from src.core.layers.feedforward import relu_ffn



class BERTEncoderBlock(nn.Module):
    '''
    post-layer_normalization 的 encoder block 架构图:
    
        ------------add------------>|                   -----add----->|
      x --bidirect_self_attention-->|--layer_norm--> x_ ---relu_ffn-->|--layer_norm--> y
   attn_mask(if any)--------------->|

    x / x_ / y have same shape
    '''
    def __init__(self, embd_size:int, num_heads:int, use_bias:bool, ffn_hidden_size:int, attn_p_drop:float, resid_p_drop:float):
        super().__init__()
        self.bidirect_attention = MultiHeadAttention(embd_size, num_heads, use_bias, attn_p_drop, resid_p_drop)
        self.layer_norm1 = nn.LayerNorm(embd_size)
        self.relu_ffn = relu_ffn(embd_size, ffn_hidden_size, resid_p_drop)
        self.layer_norm2 = nn.LayerNorm(embd_size)

    def forward(self,
                x:torch.Tensor,
                attention_mask:torch.Tensor|None = None):
        attn_result = self.bidirect_attention(x, x, x, attention_mask)
        x_ = self.layer_norm1(x + attn_result)
        y = self.layer_norm2(x_ + self.relu_ffn(x_))
        return y
