import torch
from torch import nn
import math
import torch.nn.functional as F



# 注意这里的 attn_mask 逻辑与 F.scaled_dot_product_attention 类似: True 代表合法贡献, False 代表非法要屏蔽
class ScaledDotProductAttention(nn.Module):
    '''
    args:
        attn_p_drop: dropout rate. Regularization on the attention weight matrices
        scale: scale value on the attention weight matrices
    
    inputs:
        q's shape: (B, H, L_q, d)
        k's shape: (B, H, L_kv, d)
        v's shape: (B, H, L_kv, v_size)
        attention_mask(optional)'s shape: (B, L_q, L_kv)
    
    returns: denoted as o
        o's shape: (B, H, L_q, v_size)
    
    explains:
        q(..., L_q, d), k(..., L_kv, d) --相似度计算--> attn_w(..., L_q, L_kv)
        
        Scaled Dot Production on queries & keys to attention pool for weight matrices attn_w (..., L_q, L_kv)
        Convex combinations of values based on weight matrices are returned
    
    q/k/v 都有各自的数量和维度. 其中 q 的数量自由决定, 但是其表示维度 d 要和 key 相同(因为要计算 query 和 key 之间的相似度)
    value 的维度自由决定, 但是其数量要和 key 相同(key决定了其对应value在最终输出结果中的重要程度)
    
    积式注意力 ScaledDotProductAttention 简单地根据 每条 query和不同keys之间地相似度, 决定了每个key对应的value的权重, 组合出output
    '''
    def __init__(self, attn_p_drop, scale, **kwargs):
        super().__init__(**kwargs)
        self.attn_drop = nn.Dropout(attn_p_drop)
        self.register_buffer('scale', torch.tensor(scale)) # 乘以 scale 相似度计算中因为维数过大引起的数值不稳定

    def forward(self,
                q:torch.Tensor,                             # [B, H, L_q, d]
                k:torch.Tensor,                             # [B, H, L_kv, d]
                v:torch.Tensor,                             # [B, n_kvs, v_size]
                attention_mask:torch.Tensor|None = None     # [B, L_q, L_kv] where False --> zero attend
                ):
        # q(..., L_q, d) @ k(..., L_kv, d).T -> attn_w(..., L_q, L_kv)
        attn_w = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scale
        if attention_mask is not None:
            # False 部分 not attend --> 填入大负数. 不用-inf的原因是 attention_mask 很有可能有全false行(比如存在右pad的序列)
            attn_w = attn_w.masked_fill((attention_mask.logical_not()).unsqueeze(1) , -1e20)
        
        self.attention_weights = F.softmax(attn_w, dim=-1)
        # attn_w(..., L_q, L_kv) @ v(..., L_kv, v_size) -> o(..., L_q, v_size) 
        return torch.matmul(self.attn_drop(self.attention_weights), v)



# impelementation of nn.MultiheadAttention(embd_size, num_heads, dropout=attn_p_drop, bias=use_bias, batch_first=True)
# forward(query, key, value, need_weights=False, attn_mask=attn_mask, is_causal=False)
# 注意这里 MultiHeadAttention 的 attn_mask 逻辑与 F.scaled_dot_product_attention 对齐, 即 True 代表合法贡献, False 代表非法需要屏蔽
# 而 nn.MultiheadAttention 是相反的: 该类里 attn_mask 参数 True 代表需要被屏蔽, 而 False 代表不变化
class MultiHeadAttention(nn.Module):
    def __init__(self, embd_size, num_heads, use_bias, attn_p_drop, resid_p_drop, **kwargs):
        assert embd_size % num_heads == 0, 'output dim of multihead att-pool is not divisible by number of heads'
        super().__init__(**kwargs)
        self.h = num_heads
        self.d = embd_size // num_heads
        self.W_q = nn.Linear(embd_size, embd_size, use_bias)
        self.W_k = nn.Linear(embd_size, embd_size, use_bias)
        self.W_v = nn.Linear(embd_size, embd_size, use_bias)
        self.W_o = nn.Linear(embd_size, embd_size, use_bias)
        self.attention = ScaledDotProductAttention(attn_p_drop, math.sqrt(1.0/self.d))
        self.drop = nn.Dropout(resid_p_drop)
    
    def transpose_qkv(self, x):
        '''
        x shape: (batch_size, seq_len, hidden_size)
        transpose route:--> (batch_size, hidden_size, seq_len) --> (batch_size, h, d, seq_len)
                        --> (batch_size, h, seq_len, d)
        '''
        seq_len = x.size(1)
        return x.permute(0, 2, 1).reshape(-1, self.h, self.d, seq_len).permute(0, 1, 3, 2)

    def transpose_o(self, x):
        '''
        X shape: (batch_size, h, seq_len, d)
        transpose route:--> (batch_size, h, d, seq_len) --> (batch_size, h*d, seq_len)
                        --> (batch_size, seq_len, hidden_size)
        '''
        seq_len = x.size(2)
        return x.permute(0, 1, 3, 2).reshape(-1, self.h*self.d, seq_len).permute(0, 2, 1)

    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, attention_mask:torch.Tensor|None=None):
        # q/k/v: (B, L_q/L_kv/L_kv, D)
        q_ = self.transpose_qkv(self.W_q(q)) #(B, h, L_q, d)
        k_ = self.transpose_qkv(self.W_k(k)) #(B, h, L_kv, d)
        v_ = self.transpose_qkv(self.W_v(v)) #(B, h, L_kv, d)
        o = self.attention(q_, k_, v_, attention_mask) #(B, h, L_q, d)
        output = self.transpose_o(o) #(B, L_q, D)
        return self.drop(self.W_o(output)) #(B, L_q, D)





# 因果遮罩 causal mask: 当自回归式生成 N 个 token时, 意味着用 token1 生成 token2, token1<->2 生成 token3, ..., token1<->N-1 生成 tokenN
# 1  0  0  0  0       ---> tok_2 = f(tok_1)
# 1  1  0  0  0       ---> tok_3 = f(tok_1, tok_2)
# 1  1  1  0  0       ---> tok_4 = f(tok_1, tok_2, tok_3)
# 1  1  1  1  0       ---> tok_5 = f(tok_1, tok_2, tok_3, tok_4)
# 1  1  1  1  1       ---> tok_6 = f(tok_1, tok_2, tok_3, tok_4, tok_5)
#                     ....
# 上述右边的式子, 假设 f 是线性的, 那么系数矩阵可以看作左边的因果矩阵 hadmad-乘 全系数矩阵
# 左边 因果遮罩 的构建方法:
#     最下面一行一定是 all 1, 且 1 的个数是 num all valid tokens so-far, 记为 N
#     往上相对距离为 i( 0 < i < N) 的行, causal 是 N-i 个 1, then i 个 0

# 考虑 总序列 12345, 用1234生成2345, 则 context_size = 4. 自回归系数矩阵会是一个下三角(包含对角线)矩阵
# 给 1234 的位置编码, 可以从0开始, 也可以从1开始. 只要train和infer时保持一致即可

#   1  2  3  4
# 1 a             --> 2
# 2 a  a          --> 3
# 3 a  a  a       --> 4
# 4 a  a  a  a    --> 5

# 注意力遮罩 attention mask: 注意力矩阵 [B, H, L_q, L_kv] 在乘 v[B, H, L_kv, d] 之前, 要屏蔽掉一些 非法的 贡献项.
# 这里屏蔽的意思就是: 前向计算中关键步骤的关键位置用0替代, 消除前向贡献; 反向计算中这些被替代的位置不应该参与更新其涉及的权重, 消除反向贡献.

# 首先 PAD 参与计算的就是非法贡献项
# 考虑 左PAD. 有时候有些模型会有一个序列起始符, 其实等价于在序列左端 PAD.
# 现在考虑一个 总序列12345, 用1234生成2345, 但左PAD到定长6, L_q = 6, attention mask = [001111]
# 显然有以下几个结论: 
# 一 label 序列2345也要有对应 PAD, 不能出现用 PAD 生成 TOKEN 的对齐错误. 这部分关乎 label mask, 请查阅 dataset.py 相关
# 二 attention_mask [001111] 在q行维度传播后, 可以形成x区域, 可以完美屏蔽掉 PAD 的影响.
# 给 001234 的位置编码, 由于前两个00(PAD) 会被 attention_mask 完美屏蔽, 所以它们的位置编码其实是无所谓的, 始终不会参与计算. 统一赋0就好.
# 关键还是1234的位置编码, 可以从0开始, 也可以从1开始. 只要train和infer保持一致即可. 从0开始, 全部位置编码为000123; 从1开始, 全部位置编码为001234
# 所以 左PAD情况下, 位置编码必须 只关注真实token序列1234. 所以必须是根据 attention_mask 确定真实token位置的 动态位置编码.

#   0  0  1  2  3  4
# 0 x                   --> 0
# 0 x  x                --> 0 或 1. 无论哪个, 都要被 PAD
# 1 x  x  a             --> 2
# 2 x  x  a  a          --> 3
# 3 x  x  a  a  a       --> 4
# 4 x  x  a  a  a  a    --> 5

# 现在考虑 右PAD.
# 现在考虑一个 总序列12345, 用1234生成2345, 但右PAD到定长6, L_q = 6, attention_mask = [111100]
# 发现 右PAD 的 attention mask [111100] 在q行维度传播形成的 x区域, 无法完全屏蔽掉 PAD 的影响: 在后两个 q行, 出现了以 0为q, 1234参与生成 0 的情况.
# 在给1234位置编码之后(无论0开始or1开始), 给 右PAD 的两个00的位置编码, 无论怎么赋, 都会出现在 q 中, 无法被完全屏蔽.
# 解决方法: 引入 label_mask, 即计算loss时, 屏蔽掉 label 序列中 PAD 位置. 这样 label 的 PAD位置不会 回传梯度, PAD 位置的 a 系数不会被更新
# 这样引入 label_mask 之后, 右PAD 的好处也显现出来, 即位置编码非常简单: 固定从0或1开始的序列编号即可. 不必担心给 右PAD 位置的编码, 因为会被 label_mask
# 又或者, attention_mask [111100] 不是在 q行维度简单传播, 而是具体去生成 完整遮蔽 非法区域 的 shape为 [L_q, L_kv]bool 矩阵

#   1  2  3  4  0  0
# 1 a                   --> 2
# 2 a  a                --> 3
# 3 a  a  a             --> 4
# 4 a  a  a  a          --> 5
# 0 a  a  a  a  x       --> 0 (label_mask)
# 0 a  a  a  a  x  x    --> 0 (label_mask)

# 从上述左右PAD的情况可以看出, 不管左PAD还是右PAD, 归根结底还是要尽量减少PAD. PAD 信息可以 由主动传入 attention_mask 完备表达.
# 实际上：
# 训练时 attention_mask 确实不使用简单的 q行传播, 而是根据 PAD-info / TEXT-info, 具体生成可以完整遮蔽 非法区域 的[L_q, L_q]bool 的 block-diagonal 矩阵
# 需要 attention_mask 满足: 只有 same TEXTDOC 的 tokens 才可以是合法贡献预测, 避免跨文档污染.
# label_mask 是必须的(无论左PAD还是右PAD), 因为 q -> label 必须是合法的前驱对应关系, 才可以贡献loss. 这里合法指 q 和 label same TEXTDOC.
# 推理时 attention_mask 只需要处理 PAD-info. 因为此时即使 input 为 text-packing data, 也是希望模型可以理解跨文档关系. 所以此时 attention_mask 和 position
# encoding 都应该主动透传跨文档信息, 以便模型理解.


# 通过 causal_mask/attention_mask/label_mask 三道屏蔽, 以及正确的数据对齐, 无论是左PAD还是右PAD, 在训练上差别不大. 但是在推理时, 如果作凑批Batch推理, 
# 由于始终会取Batch各序列的last token作为q, 那么右PAD的batch, 会导致某些序列在推理时传入了PAD作为q.
# 所以推理时把batch各序列作左PAD对齐 比较好. 训练数据右PAD对齐，跟推理batch左PAD对齐不矛盾.


# 参考 F.scaled_dot_product_attention 的 api 逻辑, 实现同时 带因果遮罩 causal_mask & 注意力遮罩 attention_mask 的 mha layer
# (function) def scaled_dot_product_attention( # 不支持 is_causal = True and attn_mask is not None
#     query: Tensor,
#     key: Tensor,
#     value: Tensor,
#     attn_mask: Tensor | None = None,
#     dropout_p: float = 0,
#     is_causal: bool = False,
#     scale: float | None = None,
#     enable_gqa: bool = False
# ) -> Tensor
# causal mask提供是否缓存选择:
#   缓存则 register_buffer 一个 max_context_size 的上三角 causal mask, 以便在 forward 中slice取出每条query相应的未来位置
#   不缓存则 在 forward 中按需构造一个 causal mask, 来制作每条query相应的未来位置, 这样就避免了forward中存在max_context_size
# attention中提供是否rope选择:
#   yes则在计算self-attention时, apply_rope on q/k， no则在计算self-attention时，直接计算 qk

from src.core.layers.position_encoding import RoPEConfig, RotaryPosEnc
from typing import Tuple

# 由此，CausalSelfMHA 在forward过程可以不依赖 max_context_size: causal_mask可以根据query_seq_length和kv_seq_length按需构造的, RoPE也不依赖
# 实际上，CausalSelfMHA 还实现了 input sequence 变长(前后两次forward的sequence长度可以不一样)：因为 RoPE 和 attention weights 都支持变长

# 不过, 在实际Model层面, 仍然会设定max_context_size参数，并保证所有input sequence 的长度都不大于这个值。原因如下:
#   1. 在 CausalSelfMHA层, causal_mask 和 attention weights在训练时, 涉及 O(L_q^2) 的显存空间占用. 如果不限制 L_q
#      那么在训练时若输入了过长的 input sequence as query, 可能会导致OOM. 这样在 CausalSelfMHA 层避免了过大的 attention weights 矩阵。
#   2. 很多模型在制作train dataset时，会切分长文档/拼接短文档，成固定长度的chunk。这里的固定长度就约等于模型的max_context_size，因为限制了外推能力.
#      现代很多LLM的训练策略是curriculum-style: 训练epoch早期使用短序列，后期用长序列，训练时sequence length是动态变化的。
#   3. RoPE的无限拓展只是理论上的，实际上theta=position_index*频率信号，当position很大时会失真。现代模型多会启用RoPE scaling，而这里
#      RoPE scaling 依赖 max_context_size. 为了保证 vanilla-RoPE 和 scaling-RoPE 之间接口的一致性, 应该始终保有 max_context_size 参数

class CausalSelfMHA(nn.Module):
    '''
    初始化参数
        embd_size: int
        num_heads: int
        use_bias: bool
        max_context_size: int
            最大上下文长度.
        attn_p_drop: float
            注意力权重矩阵 attention weights 在与 v 进行计算之前使用 dropout 增强泛化
        resid_p_drop: float
            因为 attention layer 后面紧接 add 残差连接. 所以要在 CausalSelfMHA layer 最后 使用 dropout 增强泛化
        use_rope: bool
            True: CausalSelfMHA layer 自带 RoPE 位置编码, 即在 qk 计算 attention weights 之前, 实施 RoPE(neox style) on q/k.
                  本 CausalSelfMHA 层在使用 RoPE 时, valid token 的位置编码从 0 开始. PAD(如果有)位置编码赋0. 不害怕混淆, 因为PAD不参与计算.
            False: CausalSelfMHA layer 不带 位置编码. 那么 CausalSelfMHA layer 之前, 要使用 max_context_size 参数实施 绝对位置编码
        use_cached_causal_mask: bool
            True: 会根据 max_context_size 预先生成 causal mask 以供裁剪去契合 attention weights
            False:, 根据 train/eval 状态, 以及 q/k 的 seq_len_q/seq_len_k, 动态生成相应 causal mask
    '''
    def __init__(self,
                 embd_size:int,
                 num_heads:int,
                 use_bias:bool,
                 max_context_size:int,
                 attn_p_drop:float,
                 resid_p_drop:float,
                 use_rope:bool,
                 use_cached_causal_mask:bool):
        
        super().__init__()
        
        assert embd_size % num_heads == 0, f'embedding size shall be divided into number_head'
        self.D = embd_size
        self.H = num_heads
        self.d = embd_size // num_heads
        assert self.d % 2 == 0, f'dim_per_head (embedding size / number_head) must be even for RoPE'
        self.scale = 1.0 / math.sqrt(self.d)

        # 合并线性映射 qkv: W_qkv(x) --> concate of [ W_q(x), W_k(x), W_v(x) ], 其中每个 W(x) 是 (B, seq_len, D) -> (B, seq_len, D)
        self.W_qkv = nn.Linear(embd_size, 3 * embd_size, bias=use_bias) # x[B, seq_len, D] --linear-proj--> qkv[B, seq_len, 3*D]
        self.W_o = nn.Linear(embd_size, embd_size, bias=use_bias) # y[B, seq_len, D] --linear-proj--> y[B, seq_len, D]
        self.attn_drop = nn.Dropout(attn_p_drop)
        self.resid_drop = nn.Dropout(resid_p_drop)

        if use_cached_causal_mask:
            # [1, 1, max_context_size, max_context_size] 的 T/F tensor. 其中主对角线上方(不含对角线)为True
            self.register_buffer(
                'causal_mask',
                torch.triu(torch.ones(max_context_size, max_context_size, dtype=torch.bool), diagonal=1)[None, None, :, :],
                persistent = False # False 指不作为 state_dict 的一部分
                )
        
        if use_rope:
            self.rope = RotaryPosEnc(RoPEConfig(self.d))

    def forward(self,
                x:torch.Tensor,                                         # [B, L_q, D]
                kv_cache:Tuple[torch.Tensor, torch.Tensor]|None = None, # [B, H, L_past, d]
                return_cache:bool = False,
                attention_mask:torch.Tensor|None = None,                # [B, L_q, L_so_far=L_past+L_q]
                positions:torch.Tensor|None = None,                     # [B, L_so_far]/[1, L_so_far]
                ):
        '''
        前向输入
        x: Tensor, shape [B, L_q, D], dim 1是 序列的排序, 即 x[:, -1, :] 代表 last token.
          train: L_q             --> 分别输入 x as first 1, 2, ..., L_q, 生成 No. 2, 3, ..., L_q+1
          infer.prefill: L_q     --> 分别输入 x as first 1, 2, ..., L_q, 生成 No. 2, 3, ..., L_q+1, 这里 No. L_q+1 是 only need
          infer.decode: L_q = 1  --> 输入 x as No. L_so_far, 配合 past as kv_cache 即 No. 1, 2, ..., L_past, 生成 No. L_so_far+1 token

        kv_cache: Tuple[Tensor, Tensor]|None, 默认None, 否则 k_cache/v_cache 分别为 tensor [B, H, L_past, d]float
          train/infer.prefill:
            x [B, L_q, D], 代表分别输入 x as first 1, 2, ..., L_q
            kv_cache = None, kv 直接使用 x 自身信息, 即 k/v as all 1<->L_q
            CauslMHA 是 自注意力计算 with (q: 1<->L_q, k: 1<->L_q, v: 1<->L_q), 生成 No. 2, 3, ..., L_q+1, 得到 attn_w[..., L_q, L_q]
            这里 attention weights[..., L_q, L_q] 还需要 经过 causal_mask / attention_mask 等, 去屏蔽 不该相关 区域.
          infer.decode:
            x [B, 1, D] 代表输入 x as No. L_so_far token.
            k_cache/v_cache 形状 [B, H, L_past, d], L_so_far = L_past+1
            k/v 使用 k_cache/v_cache 追加 x 信息, 即 k/v as all 1<->L_so_far
            CauslMHA 是计算 with (q: L_so_far, k: 1<->L_so_far, v: 1<->L_so_far), 生成 No. L_so_far+1 token, 得到 attn_w [..., 1, L_so_far].
            这里 [1, L_so_far]: 代表对于 No. L_so_far token 的 x, 生成 No. L_so_far+1 token 的系数向量. attn_mask在这里应该是全 1.

        return_cache: bool, 默认False
            当 return_cache 为 True 时, 返回计算 attention weight 时用的 kv. 这个 kv 包含 1<->L_so_far 所有.

        attention_mask: Tensor|None, 默认None, 否则为 tensor [B, L_q, L_so_far]bool.  False 代表非法贡献, True 代表合理贡献
            作用: 为 attention weights [B, H, L_q, L_so_far] 遮蔽 非法贡献. 对某 q, 涉及 PAD, 以及不与该 q same-text 的 k, 都是非法贡献, 要屏蔽.

        positions: Tensor|None, 默认None, 否则为 tensor [B, L_so_far]
            只在 use_rope = True 时, positions 才会起作用. 否则应该在 CausalSelfMHA 层之前就把 绝对位置编码 加到 x 里.
            positions 代表了 L_past + L_q = L_so_far 的所有位置信息. 从中取 后L_q部分, 即为 q 的 位置编码.
            当 use_rope = True 但又不输入 positions, 那么此 layer 会自动生成从 0 开始的 positions 作为 q/k 的位置编码.

            
        positions 和 attention_mask 应该遵循的准则: PAD position = 0; position for valid sequence starts from 0.
        positions 和 attention_mask 互为伴生, 实际上它们都应该是 PAD-info 和 TEXTDOC-info 的衍生物:
        1. PAD 位置的 position = 0; q/k 任一 PAD, attention mask = False
        2. 具备相关关系的单一序列, positions 从 0 开始从左到右递增; 序列中某位置作为 q 时, 只有具备相关关系的同序列位置 作为 k 时, attention_mask = True
        相关关系在 训练 / 推理 的定义不同. 在训练时, 只有同一 TEXTDOC 的位置(包括ENDOFTEXT)才算具备相关关系. 在推理时, 全部非PAD位置都算具备相关关系
        
        在实操中, PAD-info/TEXTDOC-info 任一或全部都可以由 segments 表示, segments 可以一起产出 positions 和 attention_mask.
        这里只考虑positions和attention_mask二者同时输入tensor(不检查是否矛盾)/二者同时为 None 两种情况:
            1.只要涉及 PAD, 那么就有 segments, 从而 positions 和 attention_mask 都有.
            2.如果 PAD-info 和 TEXTDOC-info 都没有, 那么 attention_mask 用 None 即可, positions 使用从 0 开始的递增编码即可.
        未来完成 positions/attention_mask 相互转换的函数后(elif分支), 使得本层可以处理 positions/attention_mask 恰只有其一输入的情况. 实操里不会单一输入.
        
        前向输出
            y: Tensor, shape same as x [B, L_q, D]. next-token 取 y 的 last element of dim 2.

            new_kv_cache: Tuple[Tensor, Tensor]|None. 当 return_cache == False, new_kv_cache 为 None
                train 阶段: new_kv_cache = None
                infer 阶段: new_kv_cache = tuple of k and v, k/v [B, H, L_sofar, d]. 本输出的 L_sofar 在下一次next-token生成里就是 L_past

        灵活组合 kv_cache 和 return_cache 以满足 train / infer.prefill / infer.decode 不同阶段
            train: teacher-force策略下的 self-attention 计算, 不需要输入 kv_cache(past), 同时也不需要输出 kv(so_far)
            infer.prefill: prompt 以 [B, S] 的形状输入, 不需要输入 kv_cache 因为没有. 但需要输出 kv(so_far) 给 decode 阶段.
            infer.decode: 上一次decode或者来自prefill的预测结果作为 单时间步 q 输入, 输入 kv_cache(past), 合并 q 得到 kv(so_far), 
                        输出 L_so_far+1, 并输出 kv(so_far)给下一次decode.
        '''
        L_q = x.size(1)

        # 制作 qkv: x[B, L_q, D] --> qkv[B, L_q, 3D] --split--> q/k/v[B, L_q, D] --reshape--> q/k/v[B, H, L_q, d]
        qkv = self.W_qkv(x)
        q, k, v = qkv.split(self.D, dim=-1) # q/k/v [B, L_q, D]
        # [B, L_q, D] -> [B, L_q, H, d] -> [B, H, L_q, d]
        # q/k/v 都是 H head, 此为 MHA算法. 算法 GQA/MQA 中, q和kv 有不同的 H head数量
        q = q.view(-1, L_q, self.H, self.d).transpose(1, 2) # [B, H, L_q, d]
        k = k.view(-1, L_q, self.H, self.d).transpose(1, 2) # [B, H, L_q, d]
        v = v.view(-1, L_q, self.H, self.d).transpose(1, 2) # [B, H, L_q, d]

        # only for infer.decode 阶段. 此时 L_q = 1
        if kv_cache:
            k_past, v_past = kv_cache # k_past/v_past[B, H, L_past, d]
            k, v = torch.cat([k_past, k], dim=2), torch.cat([v_past, v], dim=2) # k/v[B, H, L_so_far=L_past+L_q, d]

        L_so_far = k.size(2) # train/infer.prefill, L_q = L_so_far
        L_past = L_so_far - L_q # train/infer.prefill, L_past = 0

        if hasattr(self, 'rope'):
            if positions is not None and attention_mask is not None:
                pass # positions: [B, L_so_far], attention_mask: [B, L_q, L_so_far]
            # elif positions is not None and attention_mask is None:
            #     attention_mask = get_attention_mask_from_position_ids(positions) # [B, H, L_q, L_so_far]
            # elif positions is None and attention_mask is not None:
            #     positions = get_position_ids_from_attention_mask(attention_mask)
            else:
                positions = torch.arange(0, L_so_far, dtype=torch.long, device=k.device)
                attention_mask = None # equivalent to all True

            cos, sin = self.rope.get_sin_cos(positions, broadcast_axis=1) # cos/sin: [B/1, 1, L_so_far, d]
            cos_q, sin_q = cos[:, :, L_past:L_so_far, :], sin[:, :, L_past:L_so_far, :] # cos/sin: [B/1, 1, L_q, d]
            q = self.rope.apply_rope(q, cos_q, sin_q)
            k = self.rope.apply_rope(k, cos, sin)
        else: # positions 无用, 不关心
            if attention_mask is not None:
                pass
            # elif positions is not None:
            #     attention_mask = get_attention_mask_from_position_ids(positions)
            else:
                pass
        
        #### 等价于 y = F.scaled_dot_product_attention(q, k, v, attention_mask, attn_p_drop, True, scale=self.scale) ####
        ######### qk 计算 attn_w --> causal_mask & attention_mask --> softmax --> dropout --> attn_w @ v ---> y #########

        attn_w = torch.matmul(q, k.transpose(2, 3)) * self.scale # [B, H, L_q, L_so_far]

        # causal_mask: [B, H, L_q, L_so_far]/bool. True --> 上三角(不包含对角线, 会被-inf替代), False --> 下三角(包含对角线, 原值)
        # train/infer.prefill [..., L_q=L_so_far, L_so_far], infer.decode [..., L_q=1, L_so_far]
        if hasattr(self, 'causal_mask'):
            # 当使用缓存, 为了通用性, 采用如下写法
            causal_mask = self.causal_mask[:, :, L_past:L_so_far, :L_so_far]
        else:
            # 当动态生成
            causal_mask = torch.triu(
                torch.ones(L_q, L_so_far, dtype = torch.bool, device = attn_w.device),
                diagonal = L_past + 1
                )[None, None, :, :] # [L_q, L_so_far] --> [1, 1, L_q, L_so_far]
        attn_w = attn_w.masked_fill(causal_mask, float('-inf')) # [B, H, L_q, L_so_far]

        if attention_mask is not None:
            # attention_mask: tensor [B, L_q, L_so_far]/bool. True --> valid_area, False --> invalid_area
            # 这里用 -1e20 填入 而不再用 -inf, 因为 causal_mask 后不会有全-inf行, 可是再叠加 attention_mask 后可能会出现全 -inf 行
            # 特别是 右pad 的序列, 其attention_mask在pad位置就会出现全-inf行.
            # 全 -inf 行在 softmax 后, 全 -inf 行会变成全 nan 行, 导致计算崩溃. 因为
            # softmax计算中会用 x - x.max, 全 -inf 行的max等于 -inf, -inf-(-inx) 会出现 nan
            # 所以这里用 finite 大负数 -1e20 而不是 -inf. softmax能正确处理 大负数 和全大负数行.
            attn_w = attn_w.masked_fill((attention_mask.logical_not()).unsqueeze(1) , -1e20)

        # softmax
        attn_w = F.softmax(attn_w, dim=-1) # [B, H, L_q, L_so_far]
        attn_w = self.attn_drop(attn_w)
        y = torch.matmul(attn_w, v) # [B, H, L_q, L_so_far] @ [B, H, L_so_far, d] --> [B, H, L_q, d]

        ############ 等价部分
        # 实际上, F.scaled_dot_product_attention 在 is_causal=True时, 要求attn_mask is None, 不然报错
        # 所以若既需要 causal 又需要支持 attn_mask 时, 本实现是必不可少的

        y = y.transpose(1, 2).reshape(-1, L_q, self.D) # --> [B, L_q, H, d] --> [B, L_q, D]
        y = self.resid_drop(self.W_o(y)) # [B, L_q, D] --> [B, L_q, D]

        new_kv_cache = None
        if return_cache:
            new_kv_cache = (k, v)

        return y, new_kv_cache
    





class CausalSelfGQA(nn.Module):
    '''
    初始化参数:
        num_kv_heads: int. 当 num_kv_heads = 1 时, 即为 MQA
        其余同 CasualSelfMHA
    GQA 主要的目的是降低 kv_cache 的 size, 以降低推理时显存IO带宽压力, 另外也可以降低显存占用.
    所以有些地方用 repeat_interleave 而不是 广播 来实现 GQA, 因为认为在训练时这点显存占用不重要. 但是infer的时候肯定要用 广播机制以降低显存占用.
    '''
    def __init__(self,
                 embd_size:int,
                 num_heads:int,
                 num_kv_heads:int,
                 use_bias:bool,
                 max_context_size:int,
                 attn_p_drop:float,
                 resid_p_drop:float,
                 use_rope:bool,
                 use_cached_causal_mask:bool):
        
        super().__init__()
        
        assert embd_size % num_heads == 0, f'embedding size shall be divided into num_heads'
        assert num_heads % num_kv_heads == 0, f'num_heads shall be divided into num_kv_heads'
        self.G = num_kv_heads
        self.D = embd_size
        self.H = num_heads
        self.d = embd_size // num_heads
        assert self.d % 2 == 0, f'dim_per_head (embedding size / num_heads) must be even for RoPE'
        self.scale = 1.0 / math.sqrt(self.d)

        # 线性映射
        self.W_q = nn.Linear(embd_size, embd_size, bias=use_bias) # x[B, seq_len, D] ---> q[B, seq_len, H*d]
        self.W_kv = nn.Linear(embd_size, 2*self.G*self.d, bias=use_bias) # x[B, seq_len, D] ---> k/v[B, seq_len, G*d]
        self.W_o = nn.Linear(embd_size, embd_size, bias=use_bias) # y[B, seq_len, D] ---> y[B, seq_len, D]
        self.attn_drop = nn.Dropout(attn_p_drop)
        self.resid_drop = nn.Dropout(resid_p_drop)

        if use_cached_causal_mask:
            # [1, 1, max_context_size, max_context_size] 的 T/F tensor. 其中主对角线上方(不含对角线)为True
            self.register_buffer(
                'causal_mask',
                torch.triu(torch.ones(max_context_size, max_context_size, dtype=torch.bool), diagonal=1)[None, None, :, :],
                persistent = False # False 指不作为 state_dict 的一部分
                )
        
        if use_rope:
            self.rope = RotaryPosEnc(RoPEConfig(self.d))

    def forward(self,
                x:torch.Tensor,                                         # [B, L_q, D]
                kv_cache:Tuple[torch.Tensor, torch.Tensor]|None = None, # [B, H, L_past, d]
                return_cache:bool = False,
                attention_mask:torch.Tensor|None = None,                # [B, L_q, L_so_far=L_past+L_q]
                positions:torch.Tensor|None = None,                     # [B, L_so_far]/[1, L_so_far]
                ):
        B = x.size(0)
        L_q = x.size(1)

        # 制作 qkv: x[B, L_q, D] --> q[B, L_q, D], k/v[B, L_q, G*d] --reshape--> q[B, H, L_q, d], k/v[B, G, L_q, d]
        q = self.W_q(x)
        kv = self.W_kv(x)
        k, v = kv.split(self.G*self.d, dim=-1) # k/v[B, L_q, G*d]
        # q/k/v 都是 H head, 此为 MHA算法. 算法 GQA/MQA 中, q和kv 有不同的 head 数量: k/v 的head数量为 num_groups
        q = q.view(-1, L_q, self.H, self.d).transpose(1, 2) # [B, H, L_q, d]
        k = k.view(-1, L_q, self.G, self.d).transpose(1, 2) # [B, G, L_q, d]
        v = v.view(-1, L_q, self.G, self.d).transpose(1, 2) # [B, G, L_q, d]

        # only for infer.decode 阶段. 此时 L_q = 1
        if kv_cache:
            k_past, v_past = kv_cache # k_past/v_past[B, H, L_past, d]
            try:
                k, v = torch.cat([k_past, k], dim=2), torch.cat([v_past, v], dim=-2) # k/v[B, G, L_so_far=L_past+L_q, d]
            except RuntimeError as err:
                raise RuntimeError(f'wrong shape for k/v cache concatenation as {err}')
        
        L_so_far = k.size(2) # train/infer.prefill, L_q = L_so_far
        L_past = L_so_far - L_q # train/infer.prefill, L_past = 0

        if hasattr(self, 'rope'):
            if positions is not None and attention_mask is not None:
                pass # positions: [B, L_so_far], attention_mask: [B, L_q, L_so_far]
            # elif positions is not None and attention_mask is None:
            #     attention_mask = get_attention_mask_from_position_ids(positions) # [B, H, L_q, L_so_far]
            # elif positions is None and attention_mask is not None:
            #     positions = get_position_ids_from_attention_mask(attention_mask)
            else:
                positions = torch.arange(0, L_so_far, dtype=torch.long, device=k.device)
                attention_mask = None # equivalent to all True

            cos, sin = self.rope.get_sin_cos(positions, broadcast_axis=1) # cos/sin: [B/1, 1, L_so_far, d]
            cos_q, sin_q = cos[:, :, L_past:L_so_far, :], sin[:, :, L_past:L_so_far, :] # cos/sin: [B/1, 1, L_q, d]
            q = self.rope.apply_rope(q, cos_q, sin_q)
            k = self.rope.apply_rope(k, cos, sin)
        else: # positions 无用, 不关心
            if attention_mask is not None:
                pass
            # elif positions is not None:
            #     attention_mask = get_attention_mask_from_position_ids(positions)
            else:
                pass
        
        ## 等价于 y = F.scaled_dot_product_attention(q, k, v, attention_mask, attn_p_drop, True, scale=self.scale, enable_gqa=True) ##
        ######### qk 计算 attn_w --> causal_mask & attention_mask --> softmax --> dropout --> attn_w @ v ---> y #########

        # q[B, H, L_q, d] --> q[B, G, group_size, L_q, d], k[B, G, L_so_far, d] --> k[B, G, 1, L_so_far, d]
        # q @ k' --broadcast--> [B, G, group_size, L_q, L_so_far] --> [B, H, L_q, L_so_far]
        attn_w = torch.matmul(q.view(B, self.G, -1, L_q, self.d), k.unsqueeze(-3).transpose(-2, -1)).reshape(B, self.H, L_q, L_so_far) * self.scale

        # causal_mask: [B, H, L_q, L_so_far]/bool. True --> 上三角(不包含对角线, 会被-inf替代), False --> 下三角(包含对角线, 原值)
        # train/infer.prefill [..., L_q=L_so_far, L_so_far], infer.decode [..., L_q=1, L_so_far]
        if hasattr(self, 'causal_mask'):
            # 当使用缓存, 为了通用性, 采用如下写法
            causal_mask = self.causal_mask[:, :, L_past:L_so_far, :L_so_far]
        else:
            # 当动态生成
            causal_mask = torch.triu(
                torch.ones(L_q, L_so_far, dtype = torch.bool, device = attn_w.device),
                diagonal = L_past + 1
                )[None, None, :, :] # [L_q, L_so_far] --> [1, 1, L_q, L_so_far]
        attn_w = attn_w.masked_fill(causal_mask, float('-inf')) # [B, H, L_q, L_so_far]

        if attention_mask is not None:
            # attention_mask: tensor [B, L_q, L_so_far]/bool. True --> valid_area, False --> invalid_area
            # 这里用 -1e20 填入 而不再用 -inf, 因为 causal_mask 后不会有全-inf行, 可是再叠加 attention_mask 后可能会出现全 -inf 行
            # 特别是 右pad 的序列, 其attention_mask在pad位置就会出现全-inf行.
            # 全 -inf 行在 softmax 后, 全 -inf 行会变成全 nan 行, 导致计算崩溃. 因为
            # softmax计算中会用 x - x.max, 全 -inf 行的max等于 -inf, -inf-(-inx) 会出现 nan
            # 所以这里用 finite 大负数 -1e20 而不是 -inf. softmax能正确处理 大负数 和全大负数行.
            attn_w = attn_w.masked_fill((attention_mask.logical_not()).unsqueeze(1) , -1e20)

        # softmax
        attn_w = F.softmax(attn_w, dim=-1) # [B, H, L_q, L_so_far]
        attn_w = self.attn_drop(attn_w)
        # attn_w[B, H, L_q, L_so_far] --> [B, G, group_size, L_q, L_so_far], v[B, G, L_so_far, d] --> [B, G, 1, L_so_far, d]
        # attn_w @ v --broadcast--> [B, G, group_size, L_q, d] --> [B, H, L_q, d]
        y = torch.matmul(attn_w.view(B, self.G, -1, L_q, L_so_far), v.unsqueeze(-3)).reshape(B, self.H, L_q, self.d)

        ############ 等价部分
        # 实际上, F.scaled_dot_product_attention 在 is_causal=True时, 要求attn_mask is None, 不然报错
        # 所以若既需要 causal 又需要支持 attn_mask 时, 本实现是必不可少的

        y = y.transpose(1, 2).reshape(-1, L_q, self.D) # --> [B, L_q, H, d] --> [B, L_q, D]
        y = self.resid_drop(self.W_o(y)) # [B, L_q, D] --> [B, L_q, D]

        new_kv_cache = None
        if return_cache:
            new_kv_cache = (k, v)

        return y, new_kv_cache