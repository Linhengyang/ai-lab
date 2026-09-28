import torch
from torch import nn
from typing import Tuple
from src.core.layers.attention_pool import MultiHeadAttention, CausalSelfMHA
from src.core.layers.feedforward import relu_ffn


class TransformerDecoderBlock(nn.Module):
    def __init__(self,
                 embd_size:int,
                 num_heads:int,
                 use_bias:bool,
                 max_decoder_ctx_size:int,
                 ffn_hidden_size:int,
                 attn_p_drop:float,
                 resid_p_drop:float,
                 use_cached_causal_mask:bool
                 ):
        super().__init__()
        self.causal_attention = CausalSelfMHA(embd_size, num_heads, use_bias, max_decoder_ctx_size,
                                              attn_p_drop, resid_p_drop, False, use_cached_causal_mask)
        self.layer_norm1 = nn.LayerNorm(embd_size)
        self.cross_attention = MultiHeadAttention(embd_size, num_heads, use_bias, attn_p_drop, resid_p_drop)
        self.layer_norm2 = nn.LayerNorm(embd_size)
        self.relu_ffn = relu_ffn(embd_size, ffn_hidden_size, resid_p_drop)
        self.layer_norm3 = nn.LayerNorm(embd_size)

    def forward(self,
                x:torch.Tensor,                                         #[B, max_decoder_ctx_size/1, embd_size]
                encoded_output:Tuple[torch.Tensor, torch.Tensor],       #[B, src_ctx_size, embd_size] & [B, src_ctx_size]
                kv_cache:Tuple[torch.Tensor, torch.Tensor]|None=None,   #None / [B, H, L_past, d] & [B, H, L_past, d]
                return_cache:bool = False,
                tgt_attn_mask:torch.Tensor|None = None                  #[B, max_decoder_ctx_size, max_decoder_ctx_size] / [B, 1, L_past+1]
                ):
        # src_encoded(B, src_ctx_size, embd_size), src_attn_arr(B, src_ctx_size)
        src_encoded, src_attn_arr = encoded_output

        # causal self-attention
        attn_result, new_kv_cache = self.causal_attention(x, kv_cache, return_cache, tgt_attn_mask)
        x_ = self.layer_norm1(x + attn_result)

        # cross-attention
        tgt_attn_arr = tgt_attn_mask[:, :, 0] #[B, max_decoder_ctx_size] / [B, 1]
        cross_attn_mask = tgt_attn_arr.unsqueeze(-1) * src_attn_arr.unsqueeze(-2) #[B, max_decoder_ctx_size/1, src_ctx_size]
        cross_attn_result = self.cross_attention(x_, src_encoded, src_encoded, cross_attn_mask)
        y = self.layer_norm2(x_ + cross_attn_result)
        y_ = self.layer_norm3(y + self.relu_ffn(y))

        return y_, new_kv_cache