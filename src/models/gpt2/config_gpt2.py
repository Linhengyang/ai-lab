from dataclasses import dataclass

@dataclass
class gpt2Config:
    ## embedding layer configs
    embd_size:int
    vocab_size:int
    embd_p_drop:float
    ## decoder-block(casual_attention layer + ffn) configs
    # embd_size:int
    num_head:int
    use_bias:bool
    max_context_size:int
    attn_p_drop:float
    resid_p_drop:float
    use_cached_casual_mask:bool
    use_rope:bool
    ## number of decoder-block
    num_block:int




__all__ = ["gpt2Config"]