from dataclasses import dataclass

@dataclass
class vitConfig:
    ## pachify layer configs
    img_shape: tuple
    patch_size: int|tuple
    ## embedding config
    embd_p_drop: float
    num_hiddens: int
    ## encoder-block(bidirectional-attention+ffn) configs
    num_heads: int
    use_bias: bool
    mlp_num_hiddens: int
    resid_p_drop: float
    ## number of encoder-block
    num_blks:int


__all__ = ["vitConfig"]