import torch.nn as nn
import torch
from src.core.architectures import Encoder
from src.core.layers.position_encoding import LearnAbsPosEnc
from src.core.layers.patchify import Patchify
from src.core.layers.attention import MultiHeadAttention


class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, X):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(self.dense1(X)))))


class ViTEncoderBlock(nn.Module):
    def __init__(self, num_heads, num_hiddens, dropout, mlp_num_hiddens, use_bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(num_hiddens)
        self.attention = MultiHeadAttention(num_heads, num_hiddens, use_bias, dropout, 0.0)
        self.norm2 = nn.LayerNorm(num_hiddens)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)
    
    def forward(self, X, valid_lens=None):
        # X: flattened patches 
        # train X shape:(batch_size, num_seq, d_dim)
        # infer X shape:(batch_size, num_seq, d_dim)
        X = X + self.attention(*([self.norm1(X)]*3), valid_lens)
        
        # output shape:(batch_size, num_seq, d_dim)
        return X + self.mlp(self.norm2(X))




from .config_vit import vitConfig


class ViTEncoder(Encoder):
    def __init__(self, config:vitConfig):
        
        super().__init__()
        # patch化层: (batch_size, num_channels, height, width) --patchify--> (batch_size, num_patches, num_channels, height, width)
        # 只要 patchify 的方向是固定的, 那么对应 patch 在生成的序列的 position 也是固定的
        self.patchify = Patchify(config.img_shape, config.patch_size)
        # flatten层: (batch_size, num_patches, num_channels, height, width) --flatten from dim 2--> (batch_size, num_patches, patch_flatlen)
        # dense1层: (batch_size, num_patches, patch_flatlen) --dense--> (batch_size, num_patches, num_hiddens)
        self.dense1 = nn.Linear(self.patchify.patch_flatlen, config.num_hiddens)
        # append层: (batch_size, num_patches, num_hiddens) --append class token on dim 1 at index 0--> (batch_size, 1+num_patches, num_hiddens)
        self.register_parameter('cls_token', nn.Parameter(torch.zeros(1, 1, config.num_hiddens)))
        # position encoding层: num_hiddens = num_patches+1, 1 for class_token position
        # position embedding shape (1, 1+num_patches, num_hiddens)
        self.pos_embedding = LearnAbsPosEnc(self.patchify.num_patches+1, config.num_hiddens)
        # dropout 层:
        self.dropout = nn.Dropout(config.embd_p_drop)
        # (batch_size, 1+num_patches, num_hiddens) --ViTEncoderBlock--> (batch_size, 1+num_patches, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(config.num_blks):
            cur_blk = ViTEncoderBlock(config.num_heads, config.num_hiddens, config.resid_p_drop, config.mlp_num_hiddens, config.use_bias)
            self.blks.add_module('vitEncBlk'+str(i), cur_blk)

    def forward(self, X):
        # input shape: (batch_size, num_channels, h, w)

        # (batch_size, num_channels, h, w) --patchify with preset img_shape & patch size--> (batch_size, num_patches, num_chnls, h, w)
        # --flatten from dim 2--> (batch_size, num_patches, patch_flatlen) --dense1--> (batch_size, num_patches, num_hiddens)
        X = self.dense1( self.patchify(X).flatten(start_dim=2) )

        # cls_token (1, 1, num_hiddens) --expand batch_size at dim 0--> (batch_size, 1, num_hiddens)
        # append cls_token to data: (batch_size, num_patches, num_hiddens) --append (batch_size, 1, num_hiddens) on dim 1 at index 0-->
        # (batch_size, 1+num_patches, num_hiddens)
        X = torch.cat([self.cls_token.expand(X.shape[0], -1, -1), X], dim=1)

        position_ids = torch.arange(0, X.size(1), dtype=torch.int64, device=X.device) # 1D int64 tensor: 0, 1, ... num_patches
        # seq_len = 1+num_patches
        # X(batch_size, seq_len, num_hiddens) +(broadcast) pos_embedding(seq_len, num_hiddens)
        X = self.dropout( X + self.pos_embedding(position_ids) ) # (batch_size, seq_len, num_hiddens)
        
        # (batch_size, seq_len, num_hiddens) --> (batch_size, seq_len, num_hiddens)
        for blk in self.blks:
            X = blk(X)
        
        return X #output shape: (batch_size, seq_len, num_classes)


__all__ = ["ViTEncoder"]