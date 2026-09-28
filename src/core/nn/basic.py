# basic layers from scratch
import torch
from torch import nn
import math
import typing as t


# linear project: y = W @ x + b ---> W & b defines the projection: [out_features, in_faetures] @ [in_features,] + [out_features,] = [out_features,]
# for batch data: y_batch = x_batch @ W' + b: [..., in_features] @ [in_features, out_features] + [out_features,] = [..., out_features]
class Linear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
            )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input: torch.Tensor):
        # weight: [out_features, in_features]
        # [..., in_features] @ [in_features, out_features] + [out_features,] = [..., out_features]
        output = input @ self.weight.T
        if self.bias is not None:
            output += self.bias
        return output
    


# embedding 层: 去掉 pad / norm / scale / sparse
class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 _weight: t.Optional[torch.Tensor] = None,
                 _freeze: bool = False,
                 device=None,
                 dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # 一般情况下, 初始化 embedding 层没有指定的输入权重, 那么就随机初始化得到
        if _weight is None:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype),
                requires_grad = not _freeze # if freeze = True, then grad not required
            )
            self.reset_parameters()
        # 若 embedding 层有指定输入权重, 则需检查输入权重的shape是否相符. 不需要初始化
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = nn.Parameter(_weight, requires_grad=not _freeze)

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, input: torch.Tensor):
        # input: [...] dtype = torch.long
        # 直接索引即可
        return self.weight[input]




# dropout:
# train mode: randomly with prob p to set x_elements as 0. but re-scale expectation /(1-p)
# eval mode: nothing for x_elements
class Dropout(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                f"dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p = p
        self.inplace = inplace
    
    def forward(self,  input: torch.Tensor):
        # train mode and p > 0
        if self.training and self.p > 0:
            # prob p to drop, which means prob 1-p to keep
            keep_mask = torch.rand_like(input) > self.p
            if self.inplace:
                input *= keep_mask.float() / (1.0 - self.p) # *= 操作是 inplace . inplace 会破坏autograd图, 导致梯度报错
            else:
                input = input * keep_mask.float() / (1.0 - self.p)
            return input
        # eval mode or p = 0
        else:
            return input
        


# layernorm: x [..., (x1 ... xn)] norm on last n dims
# calculate with keepdim: mean [...,(x1 ... xn)] an& std [...,(x1 ... xn)], normalize --> x_ = (x - mean)/std as shape [...,(x1 ... xn)]
# elementwise_affine --> x_ [..., (x1 ... xn)] * w [x1 ... xn] + b [x1 ... xn]
class LayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape: t.Union[int, list[int]],
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 bias: bool = True,
                 device=None,
                 dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, device=device, dtype=dtype)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty(self.normalized_shape, device=device, dtype=dtype)
                )
            else:
                self.register_buffer('bias', None)
        else:
            self.register_buffer('weight', None)
            self.register_buffer('bias', None)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
    
    def forward(self, input: torch.Tensor):
        # 对最后 len(normalized_shape) 个维度 作 normalize, 即指定的 normalized_shape 是指倒数的 维度长度
        axis = tuple(range(-len(self.normalized_shape), 0)) # (-1,) / (-2, -1) 等

        mean = input.mean(dim=axis, keepdim=True)
        var = input.var(dim=axis, unbiased=False, keepdim=True) # layernorm 使用有偏方差, 即除以N而不是N-1
        
        # normalize
        normed = (input - mean) / torch.sqrt(var + self.eps)

        # affine
        if self.elementwise_affine:
            normed = normed * self.weight + self.bias
        
        return normed
        