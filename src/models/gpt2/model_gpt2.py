from src.core.architectures import DecoderOnly
from src.core.layers.position_encoding import LearnAbsPosEnc
from src.core.blocks.gpt2 import GPT2DecoderBlock
from src.core.functional import segments_to_positions
from src.core.generate import next_token_topk
from .config_gpt2 import gpt2Config
import torch.nn as nn
import torch
from typing import Tuple, Optional


# train mode 下, 从 input 的 segments(可以为None) 按以下标准生成对应的 positions 和 attention mask
# 若 input_segs = None, 默认 input 属于同一个text且没有PAD, 则 segments = None, positions = 从0开始总长度为L_q的位置编码 [1, L_q], attention_mask = None
# 若 input_segs != None, 那么默认其中 0 代表PAD, 位置编码赋0; 从 0 开始给每个单一 seg代表的单一序列 独立 赋位置编码, tensor [B, L_q]long
# attention_mask 是 tensor [B, L_q, L_q]long, 只有 qk 都非PAD 且 qk 的 segment 相等 才为 True, 其余都是 False
def get_segs_pos_attnmask_train(
        input_segs: None|torch.Tensor, L_q: int,
        *,
        device) -> Tuple[torch.Tensor|None, torch.Tensor, torch.Tensor|None]:
    '''
    实际的 input_segs 是 batch. e.g 只是取一行作例

    input:
        input_segs: None | tensor[B, L]long, where 0 --> PAD, 1/2/... --> TEXT ID
        e.g, [0, 0, 1, 1, 1, 2, 2]

    return:
        segments: same as input_segs
        e.g, [0, 0, 1, 1, 1, 2, 2]

        positions: tensor[B, L]long
            if input_segs is None: --> 0 to L-1 as positions
            else: position id where PAD pos --> 0, TOKEN pos --> index(starting from 0) inside every sequence
        e.g, [0, 0, 0, 1, 2, 0, 1]

        attention_mask: None | tensor[B, L, L]bool
            if input_segs is None: --> None
            else: tensor[B, L, L]bool, qk any PAD or irrelevent --> False, qk no PAD and relevent --> True
        e.g,   0  0  1  1  1  2  2 
            0  F  F  F  F  F  F  F
            0  F  F  F  F  F  F  F
            1  F  F  T  T  T  F  F
            1  F  F  T  T  T  F  F
            1  F  F  T  T  T  F  F
            2  F  F  F  F  F  T  T
            2  F  F  F  F  F  T  T
    '''
    segments = input_segs

    if segments is None:
        positions = torch.arange(0, L_q, dtype=torch.long, device=device).unsqueeze(0) # [1, L_q]
        attention_mask = None
    else: # segments [B, L]
        # train mode
        # positions: PAD pos --> 0, TOKEN pos --> index from 0 in relevent sequence
        positions = segments_to_positions(segments) # same device with segments
        # attention_mask: qk any PAD or irrelevent --> False, qk no PAD and relevent --> True
        is_pad = segments != 0 # [B, L_q]bool, PAD -> false, nonPAD -> true
        pad_mask = is_pad.unsqueeze(-1) * is_pad.unsqueeze(-2) # [B, L_q, L_q]bool, qk any PAD -> false, qk no PAD -> true
        relevent_mask = segments.unsqueeze(-1) == segments.unsqueeze(-2) # [B, L_q, L_q]bool, qk irrelevent -> false, qk relevent -> true
        attention_mask = pad_mask * relevent_mask # [B, L_q, L_q]
    
    #     [B, L_q]   [B, L_q]   [B, L_q, L_q]
    return segments, positions, attention_mask


# infer.prefill mode 下, 从 input 的 segments(可以为None) 按以下标准生成对应的 positions 和 attention mask
# 若 input_segs = None, 默认 input 属于同一个text且没有PAD, 则 segments = None, positions = 从0开始总长度为L_q的位置编码 [1, L_q], attention_mask = None
# 若 input_segs != None, 那么默认其中 0 代表PAD, 位置编码赋0; 从 0 开始给所有非 PAD 位置按序 赋位置编码, tensor [B, L_q]long
# attention_mask 是 tensor [B, L_q, L_q]long, 只有 qk 都非PAD 才为 True, 其余都是 False
def get_segs_pos_attnmask_prefill(
        input_segs: None|torch.Tensor, L_q: int,
        *,
        device) -> Tuple[torch.Tensor|None, torch.Tensor, torch.Tensor|None]:
    
    segments = input_segs

    if segments is None:
        positions = torch.arange(0, L_q, dtype=torch.long, device=device).unsqueeze(0) # [1, L_q]
        attention_mask = None
    else:
        # infer.prefill mode
        is_pad = segments != 0 # [B, L_q]bool, PAD -> false, nonPAD -> true
        # positions: PAD pos --> 0, TOKEN pos --> index from 0 in global sequence
        positions = segments_to_positions(is_pad.to(torch.long)) # same device with segments
        # attention_mask: qk any PAD --> False, qk no PAD --> True
        attention_mask = is_pad.unsqueeze(-1) * is_pad.unsqueeze(-2) # [B, L_q, L_q]bool, qk any PAD -> false, qk no PAD -> true
    
    #     [B, L_q]   [B, L_q]   [B, L_q, L_q]
    return segments, positions, attention_mask


# infer.decode mode 下, 从 input 的 input_segs(可以为None) 和 kv_cache 的 past_segs(可以为None) 按以下标准生成对应的 positions 和 attention mask
# step1: 确定 segments: 包含 past + input 的 segments
#        若 二者都为 None, 则 segments = None; 若二者有其一不为None, 则视另外一个为全1(即非PAD单一序列); 若二者皆存在, 则OK

# 若 segments = None, 则 segments = None, positions = 从0开始总长度为 L_q+L_past 的位置编码 [1, L_q+L_past], attention_mask = None
# 若 segments != None, 那么默认其中 0 代表PAD, 位置编码赋0; 从 0 开始给所有非 PAD 位置按序 赋位置编码, tensor [B, L_q+L_past]long
# attention_mask 是 tensor [B, L_q, L_q+L_past]long, 只有 qk 都非PAD 才为 True, 其余都是 False
def get_segs_pos_attnmask_decode(
        input_segs: None|torch.Tensor, L_q: int, past_segs: None|torch.Tensor, L_past: int,
        *,
        device) -> Tuple[torch.Tensor|None, torch.Tensor, torch.Tensor|None]:

    if past_segs is None and input_segs is None:
        pass
    elif past_segs is not None and input_segs is None:
        # input_segs 视作全 1. 即 nonPAD
        input_segs =  torch.ones(past_segs.size(0), L_q, dtype=torch.long, device=device)
    elif past_segs is None and input_segs is not None:
        # past_segs 视作全 1. 即 nonPAD
        past_segs = torch.ones(input_segs.size(0), L_past, dtype=torch.long, device=device)
    else:
        pass

    segments = None if past_segs is input_segs is None  else torch.cat([past_segs, input_segs], dim=-1)
    
    if segments is None:
        positions = torch.arange(0, L_past+L_q, dtype=torch.long, device=device).unsqueeze(0) # [1, L_so_far]
        attention_mask = None
    else:
        # infer.decode mode
        is_pad = segments != 0 # [B, L_so_far]bool, PAD -> false, nonPAD -> true
        # positions: PAD pos --> 0, TOKEN pos --> index from 0 in global sequence
        positions = segments_to_positions(is_pad.to(torch.long)) # same device with segments
        # attention_mask: qk any PAD --> False, qk no PAD --> True
        is_q_pad = input_segs != 0 # [B, L_q]bool, PAD -> false, nonPAD -> true
        attention_mask = is_q_pad.unsqueeze(-1) * is_pad.unsqueeze(-2) # [B, L_q, L_so_far]]bool, qk any PAD -> false, qk no PAD -> true
    
    # [B,L_so_far] / [B,L_so_far] / [B,L_q,L_so_far]
    return segments, positions, attention_mask







class gpt2Model(DecoderOnly):
    def __init__(self, config:gpt2Config):
        super().__init__()
        self.config = config

        self.W_tok_embd = nn.Embedding(config.vocab_size, config.embd_size)
        if not config.use_rope:
            self.W_pos_embd = LearnAbsPosEnc(config.max_context_size, config.embd_size) # 可学习绝对位置编码自带随机初始化
        self.embd_drop = nn.Dropout(config.embd_p_drop)

        self.blocks = nn.ModuleList([
            GPT2DecoderBlock(
                config.embd_size,
                config.num_head,
                config.use_bias,
                config.max_context_size,
                config.attn_p_drop,
                config.resid_p_drop,
                config.use_rope,
                config.use_cached_casual_mask) for _ in range(config.num_block)])
        
        self.layer_norm_final = nn.LayerNorm(config.embd_size, bias=config.use_bias)

        # tied weight with embedding layer: 可以看作 隐藏状态和 token_embd 的语义相似度计算, 也可以视作 正则化手段防止过拟合
        self.head_tok = nn.Linear(config.embd_size, config.vocab_size, bias=False)
        self.head_tok.weight = self.W_tok_embd.weight # (vocab_size, embd_size). 这行代码本质上将两个weight指向了同一块内存区域

        self.apply(self._init_weights) # _init_weights 中对 linear/embedding 的weights 作相同分布的初始化. 由于已tied, 故两层都以最后一次初始化为结果


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self,
                input_seqs: torch.Tensor,
                input_segs: Optional[torch.Tensor] = None,
                past_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
                past_segs: Optional[torch.Tensor] = None,
                if_cache_kv: bool = False
                ):
        '''
        input:
            input_seqs: tensor [B, L_q]long whose value must within embedding table
            input_segs: None | tensor [B, L_q]long to describe input_seqs' PAD-info(0 for PAD) & TEXT-info(1/2...for textID)
            past_kv: None | tuple of past(k_cache, v_cache), k_cache/v_cache tensor [B, H, L_past, d]float
            past_segs: None | tensr [B, L_past]long to describe past_kv's PAD-info & TEXT-info
            if_cache_kv: bool to determine if to return updated k_cache/v_cache as in tuple

        output:
            logits: [B, L_q, vocab_size]
            past_kv: None | tuple of so_far(k_cache, v_cache). for next-time, it'll be past
        
        本模型默认准则: segment 中, 0 代表 PAD, PAD 的 segment 赋 0; 位置编码中, PAD 位置编码赋 0, 有效序列的位置编码从 0 开始(第一位赋0)
        本模型对 input 序列中, PAD 具体赋哪个值 并不关心(因为 PAD 会被 attention_mask 屏蔽计算). 但是此值必须在embedding table范围内.
        数据处理时, input 序列中的 PAD 位置可用 0 直接填充, 这样永远不会 out-of-table, 处理也方便.
        '''
        B, L_q = input_seqs.shape
        device = input_seqs.device
        L_past = past_kv[0][0].size(2) if past_kv is not None else 0

        if self.training:
            segments, positions, attention_mask = get_segs_pos_attnmask_train(input_segs, L_q, device=device)
        elif past_kv is None:
            segments, positions, attention_mask = get_segs_pos_attnmask_prefill(input_segs, L_q, device=device)
        else:
            segments, positions, attention_mask = get_segs_pos_attnmask_decode(input_segs, L_q, past_segs, L_past, device=device)
        
        tok = self.W_tok_embd(input_seqs) # [B, L_q] -> [B, L_q, D]

        # 如果存在绝对位置编码层: 要 add abs pos encoding 到 tok embedding 上
        if hasattr(self, 'W_pos_embd'):
             # positions [B, L_so_far=L_past + L_q], 取 last L_q 列
            positions_q = positions[:, -L_q:] # [B, L_so_far] --> [B, L_q]
            
            tok = tok + self.W_pos_embd(positions_q) # [B, L_q, D]
            positions = None # 位置编码已经加到 tok embedding 里了, 不再使用
        
        # 如果使用RoPE位置编码: 无需额外操作, casual attention 层会执行RoPE
        x = self.embd_drop(tok) # [B, L_q, D]
        
        new_past_kv = [] if if_cache_kv else None
        for i, block in enumerate(self.blocks):
            kv_cache = past_kv[i] if past_kv is not None else None
            # new_kv_cache 是 kv_cache concate k & v from x
            x, new_kv_cache = block(x, kv_cache, if_cache_kv, attention_mask, positions)
            if if_cache_kv:
                new_past_kv.append( new_kv_cache ) # new_kv_cache 是 torch.cat 得到的, 其内存使用是高效的.
        
        logits = self.head_tok(self.layer_norm_final(x)) # [B, L_q, vocab_size]

        return logits, tuple(new_past_kv) if if_cache_kv else None, segments

    @torch.no_grad()
    def generate(self,
                 input_ids: torch.Tensor,           # prefill: [B, L_q], decode: [B, 1]
                 input_segs: torch.Tensor|None,     # [B, L_q]
                 max_gen_size: int,                 # max generating length
                 temperature: float = 1.0,          # flatten/sharpen the output distribution
                 top_k: int|None = None,            # limit selections when sampling token via output distribution
                 eos_id: int|None = None            # stop sign
                 ):
        '''
        generate 分为两个阶段:
        1. prefill: input_ids [B, S]
        此时, forward 输入:
            input_seqs = input_ids [B, S]
            input_segs 根据 input_seqs 的实际 PAD/TEXT 情况输入, None/[B, S]. 实际中 infer 中不再屏蔽跨文档相关性, 仅屏蔽PAD
            past_kv = None 因为此时没有 past
            past_segs = None 因为此时没有 past
            if_cache_kv = True 因为在 generate 时, 总需要把该次 so-far kv 作为下一次 infer 的 past_kv 输入
        prefill 的输出 output [B, S]. 取出 output 的 last dim 2 --> [B, 1] 作为输入进入阶段2

        2. decode: input_ids [B, 1]
        此时, forward 输入:
            input_seqs = input_ids [B, 1]
            input_segs = None 即使产出 eos 也不算 PAD, 希望模型考虑 eos 而不是屏蔽 eos
            past_kv = tuple of k/v [B, H, L_past, d], 即上一次 generate 返回的 so-far kv. 上一次的so-far是这一次的past
            past_segs: None/[B, L_past], 即上一次 generate 返回的 so-far segments. 上一次的so-far是这一次的past
            if_cache_kv = True 因为在 generate 时还是要保持输出该次 so-far kv 作为下一次 infer 的 past_kv 输入
        '''
        assert max_gen_size > 0 and max_gen_size + input_ids.size(1) <= self.config.max_context_size, \
            f'max_gen_size must be larger than 0 and not larger than max_context_size minus prompt sequence length'
        self.eval()

        # prefill
        logits, past_kv, past_segs = self(
            input_seqs = input_ids,             # [B, L_q]
            input_segs = input_segs,            # [B, L_q]|None
            past_kv = None,
            past_segs = None,
            if_cache_kv = True
        )
        # logits: [B, S, vocab_size]
        # past_kv: tuple of kv_cache [B, H, L_past=L_q, d]
        # past_segs: describe of past kv [B, L_past=L_q]|None

        max_gen_size -= 1
        next_token = next_token_topk(logits[:, -1, :], temperature, top_k) # [B, 1]
        output = next_token.cpu() # [B, 1]

        # 若提供了 EOS, 则检查输出结果是否全 EOS. 若是, 则退出 generate
        if eos_id is not None and (next_token == eos_id).all():
            return output
        
        # decode
        for _ in range(max_gen_size):
            logits, past_kv, past_segs = self(
                input_seqs = next_token,                    # [B, 1]
                input_segs = None,                          # None
                past_kv = past_kv,                          # tuple of kv_cache [B, H, L_past, d]
                past_segs = past_segs,                      # [B, L_past]|None
                if_cache_kv = True)

            # logits: [B, 1, vocab_size]
            # past_kv: tuple of kv_cache [B, H, L_past+1, d]
            # past_segs: describe of past kv [B, L_past+1] | None
            next_token = next_token_topk(logits.squeeze(1), temperature, top_k) # [B, 1]
            output = torch.cat([output, next_token.cpu()], dim=-1)

            # 若提供了 EOS, 则检查输出结果是否全 EOS. 若是, 则退出 generate
            if eos_id is not None and (next_token == eos_id).all():
                return output
        
        return output # [B, max_gen_size]



__all__ = ["gpt2Model"]