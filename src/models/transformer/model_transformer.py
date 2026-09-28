from src.core.architectures import EncoderDecoder
from src.core.layers.position_encoding import TrigonoAbsPosEnc
from src.core.blocks.bert import BERTEncoderBlock
from src.core.blocks.transformer import TransformerDecoderBlock
from src.core.generate import next_token_topk
from .config_transformer import transformerConfig
import torch.nn as nn
import math
import torch
from typing import Optional, Tuple



class TransformerEncoder(nn.Module):
    def __init__(self, config:transformerConfig):
        super().__init__()

        self.D = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_encoding = TrigonoAbsPosEnc(config.hidden_size)
        self.embd_drop = nn.Dropout(config.embd_p_drop)

        self.blks = nn.Sequential()
        for i in range(config.num_enc_blks):
            cur_blk = BERTEncoderBlock(
                config.hidden_size,
                config.num_heads,
                config.use_bias,
                config.ffn_hidden_size,
                config.attn_p_drop,
                config.resid_p_drop,
                )
            self.blks.add_module(f'enc_blk{i+1}', cur_blk)
    
    def forward(self,
                src: torch.Tensor,
                src_valid_lens: torch.Tensor
                ):
        # src shape: (batch_size, src_context_size)int64
        # src_valid_lens: (batch_size,)int64

        src_context_size = src.size(1)
         # 使用 固定位置编码时, 避免位置编码的影响过大，所以放大input embeddings
        src_embd = self.token_embedding(src)*math.sqrt(self.D) # (batch_size, src_context_size, hidden_size)

        # src position embedding: 没有 bos, 从 1 开始到 src_context_size
        position_ids = 1 + torch.arange(0, src_context_size, dtype=torch.int64, device=src_embd.device)
        src_embd = self.embd_drop( src_embd + self.pos_encoding(position_ids) ) # (batch_size, src_context_size, hidden_size)

        # attention mask(batch_size, src_context_size, src_context_size): True --> valid area, False --> need masked
        # 没有因果自回归, 只有 位置id大于 valid_lens 部分为 invalid area(PAD) --> False, 位置id小于等于 valid_lens 部分为 valid area(non-PAD) --> True
        src_attn_arr = position_ids[None, :] <= src_valid_lens[:, None] # (batch_size, src_context_size)
        src_attn_mask = src_attn_arr.unsqueeze(-1) * src_attn_arr.unsqueeze(-2)
        for blk in self.blks:
            src_embd = blk(src_embd, src_attn_mask)

        return src_embd, src_attn_arr # (batch_size, src_context_size, hidden_size) & (batch_size, src_context_size)
    



class TransformerDecoder(nn.Module):
    def __init__(self, config:transformerConfig):
        super().__init__()

        self.D = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_encoding = TrigonoAbsPosEnc(config.hidden_size)
        self.embd_drop = nn.Dropout(config.embd_p_drop)

        self.blks = nn.Sequential()
        for i in range(config.num_dec_blks):
            cur_blk = TransformerDecoderBlock(
                config.hidden_size,
                config.num_heads,
                config.use_bias,
                config.max_decoder_ctx_size,
                config.ffn_hidden_size,
                config.attn_p_drop,
                config.resid_p_drop,
                config.use_cached_causal_mask
                )
            self.blks.add_module(f'dec_blk{i+1}', cur_blk)
        
        # tied weight with embedding layer: 可以看作 隐藏状态和 token_embd 的语义相似度计算, 也可以视作 正则化手段防止过拟合
        self.head_tok = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.head_tok.weight = self.token_embedding.weight # 这行代码本质上将两个weight指向了同一块内存区域

        self._decoder_context_size = config.max_decoder_ctx_size # decode 支持的最大 context size 作为重要参数透出


    def forward(self,
                encoded_output:Tuple[torch.Tensor, torch.Tensor],                       # [B, src_ctx_size, hidden_size]/[B, src_ctx_size]
                tgt: torch.Tensor,                                                      # [B, max_decoder_ctx_size]
                tgt_valid_lens: Optional[torch.Tensor] = None,                          # [B, ]
                past_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,# tuple of (past_k[B, H, L_past, d], past_v[B, H, L_past, d])
                if_cache_kv: bool = False
                ):
        
        if self.training:
            # train: tgt 是 [B, max_decoder_ctx_size], tgt_valid_lens 是 [B, ]
            positions = torch.arange(0, tgt.size(1), dtype=torch.int64, device=tgt.device) # [max_decoder_ctx_size, ]
            tgt_attn_arr = positions[None, :] < tgt_valid_lens[:, None] # [B, max_decoder_ctx_size]
            tgt_attn_mask = tgt_attn_arr.unsqueeze(-1) * tgt_attn_arr.unsqueeze(-2) #[B, max_decoder_ctx_size, max_decoder_ctx_size]
        elif past_kv is None:
            # infer on <bos>: tgt 是[B, 1]的<bos>tensor, tgt_valid_lens 不需要输入
            positions = torch.tensor([0], dtype=torch.int64, device=tgt.device)
            tgt_attn_mask = torch.ones((tgt.size(0), 1, 1), dtype=torch.bool, device=tgt.device) #[B, 1, 1]
        else:
            L_past = past_kv[0][0].size(2)
            # infer on last predicted token: tgt 是[B, 1], tgt_valid_lens 不需要输入
            positions = torch.tensor([L_past], dtype=torch.int64, device=tgt.device)
            tgt_attn_mask = torch.ones((tgt.size(0), 1, L_past+1), dtype=torch.bool, device=tgt.device) #[B, 1, L_past+1]
        
        x = self.token_embedding(tgt)*math.sqrt(self.D) # (batch_size, max_decoder_ctx_size/1, hidden_size)
        x = self.embd_drop( x + self.pos_encoding(positions) ) # (batch_size, max_decoder_ctx_size/1, hidden_size)

        new_past_kv = [] if if_cache_kv else None
        for i, block in enumerate(self.blks):
            kv_cache = past_kv[i] if past_kv is not None else None
            # new_kv_cache 是 kv_cache concate k & v from x
            x, new_kv_cache = block(x, encoded_output, kv_cache, if_cache_kv, tgt_attn_mask)
            if if_cache_kv:
                new_past_kv.append( new_kv_cache ) # new_kv_cache 是 torch.cat 得到的, 其内存使用是高效的
        
        logits = self.head_tok(x) # [B, max_decoder_ctx_size/1, vocab_size]
        return logits, tuple(new_past_kv) if if_cache_kv else None


    @property
    def decoder_context_size(self):
        return self._decoder_context_size



class Transformer(EncoderDecoder):
    def __init__(self, encoder: TransformerEncoder, decoder: TransformerDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_valid_lens: torch.Tensor,
                tgt_valid_lens: torch.Tensor|None = None,
                past_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
                if_cache_kv: bool = False
                ):
        encoded_outputs = self.encoder(src, src_valid_lens)
        return self.decoder(encoded_outputs, tgt, tgt_valid_lens, past_kv, if_cache_kv)

    @torch.no_grad()
    def generate(self,
                 src_ids: torch.Tensor,         # [B, L_src]
                 src_valid_lens: torch.Tensor,  # [B,]
                 max_gen_size: int,             # max generating length
                 bos_id: int,                   # begin of sequence sign
                 eos_id: int,                   # end of sequence sign
                 temperature: float = 1.0,      # flatten/sharpen the output distribution
                 top_k: int|None = None         # limit selections when sampling token via output distribution
                 ):
        assert max_gen_size > 0 and max_gen_size <= self.decoder.decoder_context_size, \
            f'max_gen_size must be larger than 0 and no larger than decoder_context_size'
        self.eval()

        # src encode
        encoded_outputs = self.encoder(src_ids, src_valid_lens)

        # generate on <bos>: [B, 1]
        tgt = torch.empty((src_ids.size(0), 1), device=src_ids.device, dtype=torch.int64).fill_(bos_id)

        logits, past_kv = self.decoder(
            encoded_outputs,
            tgt,
            past_kv = None,
            if_cache_kv = True
            )
        # logits: [B, 1, vocab_size]
        # past_kv: tuple of kv_cache [B, H, L_past=1, d]

        max_gen_size -= 1
        next_token = next_token_topk(logits[:, 0, :], temperature, top_k) # [B, 1]
        output = next_token.cpu() # [B, 1]

        # 检查输出结果是否全 EOS. 若是, 则退出 generate
        if (next_token == eos_id).all():
            return output
        
        # generate
        for _ in range(max_gen_size):
            logits, past_kv = self.decoder(
                encoded_outputs,
                tgt = next_token,                   # [B, 1]
                past_kv = past_kv,                  # tuple of kv_cache [B, H, L_past, d]
                if_cache_kv = True
                )

            # logits: [B, 1, vocab_size]
            # past_kv: tuple of kv_cache [B, H, L_past+1, d]
            next_token = next_token_topk(logits.squeeze(1), temperature, top_k) # [B, 1]
            output = torch.cat([output, next_token.cpu()], dim=-1)

            # 输出结果是否全 EOS. 若是, 则退出 generate
            if (next_token == eos_id).all():
                return output
        
        return output # [B, max_gen_size] on cpu





__all__ = ["TransformerEncoder", "TransformerDecoder", "Transformer"]