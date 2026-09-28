from dataclasses import dataclass

@dataclass
class transformerConfig:
    # embd params
    vocab_size: int
    hidden_size: int
    num_heads: int
    use_bias: bool
    ffn_hidden_size: int
    # dropout params
    embd_p_drop: float
    attn_p_drop: float
    resid_p_drop: float
    # encoder params
    num_enc_blks: int
    # decoder params
    num_dec_blks: int
    max_decoder_ctx_size: int
    use_cached_causal_mask: bool




__all__ = ["transformerConfig"]