import torch.nn as nn
import torch
import math
from src.core.layers.attention import MultiHeadAttention
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





from src.core.layers.position_encoding import LearnAbsPosEnc, TrigonoAbsPosEnc
from .config_bert import bertConfig

# Encoder 组件: 所有预训练 encoder 组件应该干同一样事情: 把shape为 (batch_size, seq_length) 的序列数据, 转换为 (batch_size, seq_length, hidden_size)

class BERTEncoder(nn.Module):
    def __init__(self, config:bertConfig):
        super().__init__()
        # token embedding layer: (batch_size, seq_length)int64 of vocab_size --embedding--> (batch_size, seq_length, hidden_size)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # position embedding layer: (seq_length, )int64 of position ID --embedding--> (seq_length, hidden_size)
        if config.use_abspos:
            self.pos_encoding = TrigonoAbsPosEnc(config.hidden_size)
        else:
            self.pos_encoding = LearnAbsPosEnc(config.seq_len, config.hidden_size)
        # 输入序列已经被 pad/truncate 到同一长度, 并且额外 append 的 <cls> 和 <sep> token 也被计算在 seq_len 里了
        
        # embd_drop layer: dropout on token_embd + pos_embd
        self.embd_drop = nn.Dropout(config.embd_p_drop)

        # segment embedding layer: (batch_size, seq_length)int64 of 0/1 --embedding--> (batch_size, seq_length, hidden_size)
        self.seg_embedding = nn.Embedding(2, config.hidden_size)

        # encoder layer: token embd + pos embd + seg embd: (batch_size, seq_length, hidden_size) --single encoder block-->
        # (batch_size, seq_length, hidden_size) 
        self.blks = nn.Sequential()
        for i in range(config.num_blks):
            cur_blk = BERTEncoderBlock(
                config.hidden_size,
                config.num_heads,
                config.use_bias,
                config.ffn_hidden_size,
                config.attn_p_drop,
                config.resid_p_drop,
                )
            self.blks.add_module(f'blk{i+1}', cur_blk)
    

    def forward(self,
                tokens: torch.Tensor,
                valid_lens: torch.Tensor,
                segments: torch.Tensor
                ):
        # tokens shape: (batch_size, seq_len)int64
        # valid_lens: (batch_size,)
        # segments shape: (batch_size, seq_len)01 int64

        seq_len = tokens.size(1)
        input_embd = self.token_embedding(tokens) + self.seg_embedding(segments) # X shape: (batch_size, seq_len, hidden_size)

        # positions shape: [0, 1, ..., seq_len-1] 1D tensor
        position_ids = torch.arange(0, seq_len, dtype=torch.int64, device=input_embd.device)
        # input_embd(batch_size, seq_len, hidden_size) broadcast + posenc(seq_len, hidden_size)
        input_embd = self.embd_drop( input_embd + self.pos_encoding(position_ids) ) #(batch_size, seq_len, hidden_size)

        # attention mask(batch_size, seq_len, seq_len): True --> valid area, False --> need masked
        # 没有因果自回归, 只有 invalid area(PAD) --> False, valid area(non-PAD) --> True
        attn_mask = position_ids[None, :] < valid_lens[:, None] # (batch_size, seq_len)
        attn_mask = attn_mask.unsqueeze(-1) * attn_mask.unsqueeze(-2) # (batch_size, seq_len, seq_len)

        for blk in self.blks:
            input_embd = blk(input_embd, attn_mask)

        return input_embd # 输出 shape: (batch_size, seq_len, hidden_size)
    


__all__ = ["BERTEncoder"]