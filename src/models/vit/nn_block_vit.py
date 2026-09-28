from torch import nn
from src.core.layers.attention_pool import MultiHeadAttention

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