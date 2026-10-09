# Tokenizer.py
# new version of Tokenizer inspired by GPT tokenizer(tiktokenizer)

from abc import ABC
import typing as t


class Tokenizer(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def encode(self, string: str) -> t.List[int]:
        raise NotImplementedError
    
    def decode(self, indices: t.List[int]) -> str:
        raise NotImplementedError



class CharacterTokenizer(Tokenizer):
    '''
    A tokenizer mapping between unicode characters and its unicode number as tokens
    encode: char --> integer between 0 and 0x10FFFF
    decode: integer between 0 and 0x10FFFF --> char
    '''
    def encode(self, string:str) -> t.List[int]:
        return list( map(ord, string) )
    
    def decode(self, indices: t.List[int]) -> str:
        # filter valid unicode index
        indices = [ i  for i in indices if 0<= i <= 0x10FFFF ]
        return "".join(map(chr, indices))
    


class ByteTokenizer(Tokenizer):
    '''
    A tokenizer splits string into bytes, and use integers between 0 and 255 as tokens
    encode: string --> integer(between 0 and 255) sequence
    decode: integer(between 0 and 255) sequence --utf-8--> string with possible replacement
    '''
    def encode(self, string: str) -> t.List[int]:
        string_bytes = string.encode("utf-8") # 返回 utf-8 规范编码的 字节byte 序列. 所以返回的 string_bytes 是一个序列
        # 可以求 len, 返回 字节数量; 可以索引 index，返回各个 字节的整数值(0-255)
        # 英文 1 字节，欧洲字符 2字节，中日韩字符 3字节，罕见字符 4字节
        indices = list( map(int, string_bytes) ) # list of integers btw 0-255
        return indices
    
    def decode(self, indices: t.List[int]) -> str:
        # filter valid unicode index
        try:
            string_bytes = bytes(indices) # bytes 其中一种使用方式: 输入 list of integers, 要求每个 integer 0-255
            return string_bytes.decode('utf-8', errors='replace')
        except ValueError:
            print(f'input indices {indices} has wrong values. must be 0-255')






import regex as re
import os
import functools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from ..math import check_monotonic


GPT4_TOKENIZER_REGEX = \
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


ENDOFTEXT = '<|endoftext|>'
FIM_PREFIX = '<|fim_prefix|>'
FIM_MIDDLE = '<|fim_middle|>'
FIM_SUFFIX = '<|fim_suffix|>'
ENDOFPROMPT = '<|endofprompt|>'




def get_pair_counts(tokens:t.List[int|str], p_counts:t.Dict[tuple[int|str, int|str], int]|None = None):
    '''
    p_counts 如果是 None: init a p_counts and return
        给定 tokens list, 计算它的 pair-tokens counts, 返回一个 pair counts dict
    p_counts 如果不是 None: in-place update p_counts
        给定 tokens list, 计算它的 pair-tokens counts, 更新 到 输入的 p_counts 里
    输入的 tokens 可以是 list of int, 也可以是 list of str. 因为 token 可以由 int / str 两种方式表示.
    '''
    if p_counts is None:
        p_counts = {}
    if len(tokens) == 1: # 如果只剩下 一个 token, 那么就无需 count pair / 更新 p_counts 
        return p_counts
    
    for pair in zip(tokens, tokens[1:]):
        p_counts[pair] = p_counts.get(pair, 0) + 1

    return p_counts





def merge_pair(tokens:t.List[int|str], pair:tuple[int|str, int|str], new_token:int|str) -> t.List[int|str]:
    new_tokens = [] # 返回一个 new_tokens, 而不是对 tokens 作 in-place 更新
    i = 0
    if len(tokens) == 1: # 如果只剩下 一个 token, 那么就没有 merge pair 的必要
        return tokens
    
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            new_tokens.append( new_token )
            i += 2
        else:
            new_tokens.append( tokens[i] )
            i += 1
    return new_tokens





# 缓存这个函数, 因为预计会用完全相同的输入，调用很多次这个函数。缓存函数调用结果，下次遇到相同输入时，避免计算直接返回缓存结果
@functools.lru_cache(maxsize=128)
def _special_token_regex(tokens: frozenset[str]) -> "re.Pattern[str]":
    inner = "|".join(re.escape(token) for token in tokens)
    # 返回诸如 "(<pad>|<eos>|<bos>)" 之类的 special tokens 或运算符 拼接的 compiled regex
    # compiled regex 可以直接调用 .search(text) 方法
    return re.compile(f"({inner})")



def raise_run_out_corpus_error(num_occured_merges:int, num_specials:int) -> t.NoReturn:
    '''
    如果经过 tokens 累积更新 p_counts, p_counts 仍然是 {}, 说明 corpus 已经全部 merge 到一起, 无可 merge。
    不提前结束 merge 循环, 而是报错, 提示 更换参数 explicit_n_vocab 或 语料 corpus。
    原因是参数 explicit_n_vocab 语意为「明确的 vocab_size」,
    所以不应引入不确定性：当 explicit_n_vocab 和 corpus 冲突时, raise error而不是根据 corpus 确定 explicit_n_vocab。
    '''
    raise RuntimeError(
        f'run out of corpus(all merged together) after {num_occured_merges} merges.\n'
        f'the maximum valid `explicit_n_vocab` for this corpus is {256+num_occured_merges+num_specials}.\n'
        f're-init the tokenizer with lower `explicit_n_vocab` in between {256+num_specials}'
        f'(zero-merge) & {256+num_occured_merges+num_specials}(exactly-ran-out-of current corpus), '
        f'or with enlarged corpus.'
        )






def raise_disallowed_special_token(token: str) -> t.NoReturn:
    raise ValueError(
        f'disallowed specials {token!r} found in text.\n'
        f'expand `allowed_special` if you want to encode the disallowed marks into special tokens.\n'
        f'narrow `disallowed_special` if you want to encode the disallowed marks as plain.\n'
        f'you can expand `allowed_special` to "all" or narrow `disallowed_special` to (),'
        f'both will ignore specials and tokenize the text as plain'
        )






def encode_to_ints(s:str, encoding='utf-8') -> t.List[int]:
    return list( s.encode(encoding) )



class baseBBPETokenizer(Tokenizer):
    '''
    merge_ranks 是 dict of [token_L, token_R] ---> merged_token
    其中 merged_token 是从 256 开始编号, 即 rank + 256, rank = 0, ..., num_merges-1
    故 rank(等同 merge_rank)是0开始的、merged_token 相对 256 的偏移量
    '''
    def __init__(
            self,
            name: str,
            buffer_dir: str,
            pat_str: str = GPT4_TOKENIZER_REGEX,
            merge_ranks: dict[tuple[int, int], int] = {},
            special_marks: list[str] = [ENDOFTEXT, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, ENDOFPROMPT],
            explicit_n_vocab: int | None = None,
            **kwargs):

        self.name = name
        self._buffer_dir = buffer_dir
        # set the parquet dataset/file directories
        os.makedirs(self._buffer_dir, exist_ok=True)
        self.pat_str = pat_str
        self._merge_ranks = merge_ranks
        self._special_marks = special_marks
        # special marks 必须都能被 pat_str 切开，不然可能会导致 merge-regenerate-special_mark in BPE, 导致混淆
        assert all([ len(re.findall(pat_str, mark)) > 1 for mark in special_marks ]) # special_marks 可以为 []

        if merge_ranks: # 如果输入了非空的 merge_ranks
            # merge_ranks 的 RANK 应该是 256 至 MAX_RANK 的连续正整数: 递增 且 len(merge_ranks) = MAX_RANK - 255 且 首元素 = 256
            ranks_seq = list( merge_ranks.values() )
            assert check_monotonic(ranks_seq, mode='increase', strict=True) and len(ranks_seq) == ranks_seq[-1]-255 and ranks_seq[0] == 256

            # 可以直接 构建 vocab: token_ID --> bytes
            self._build_vocab()
            
            # 可以直接 注册 special_tokens，因为已经有 merge_ranks，无需 BPE train
            self._register_special_tokens()

            # 总的 vocab_size, 即 explicit_n_vocab 也随之 确定。若输入了 explicit_n_vocab，检查是否和 merge+special 匹配
            if explicit_n_vocab:
                # 总 vocab size 应该等于 256 + merge tokens size + n_special_tokens
                assert explicit_n_vocab == 256 + len(merge_ranks) + len(special_marks)

            self.explicit_n_vocab = 256 + len(merge_ranks) + len(special_marks)

        else: # 如果 没有输入非空的 merge_ranks
            # 那么需要 run BPE train process to build merges_ranks forrest. corpus text 将随后输入
            # 如果输入了 explicit_n_vocab, 只要 valid, 那么总 explicit_n_vocab将在这里确定
            if isinstance(explicit_n_vocab, int):
                assert explicit_n_vocab >= 256 + len(special_marks), \
                    f'pretrained merge_ranks forrest empty.\ninput explicit_n_vocab (shall be at least greater ' + \
                    f'than 256+{len(special_marks)}(num_special_marks)={256+len(special_marks)}.\n' + \
                    f'e.g, GPT2 tokenizer has explicit_n_vocab as 50257, with 1 special marks and 50000 merges.'
                
                self.explicit_n_vocab = explicit_n_vocab



    def _build_vocab(self):
        # vocab: token_ID --> bytes
        assert hasattr(self, "_merge_ranks")
        self._vocab = {i: bytes([i]) for i in range(256)} # initialize 0 - 255 --> bytes
        # _merge_ranks 必须是按 BPE 生成序(即256开始递增) 插入字典的. .item() 方法能保证插入序返回 key 和 value
        for (L_int, R_int), merged_int in self._merge_ranks.items():
            self._vocab[merged_int] = self._vocab[L_int] + self._vocab[R_int] # two bytes concatenate



    def _register_special_tokens(self):
        assert hasattr(self, "_merge_ranks") and hasattr(self, "_special_marks")
        # vocab key 是 int, value 是 bytes
        # inverse special tokens key 是 int, value 是 str
        # special tokens key 是 str, value 是 int
        # 不统一，是因为 special tokens 要保持 str, 以便作 正则分割操作
        self.special_tokens: dict[str, int] = {} # speical mark str --> special token id

        # special tokens 的 token ID 应该 紧接着 merge_ranks 的 MAX RANK，即 MAX_RANK + 1 开始
        # 这样 所有tokens的 mapping value 应该是 0 至 explicit_n_vocab-1 = 256 + len(merge_ranks) + len(special_marks) - 1 的连续正整数
        # 所以 注册 special_tokens 的工作应该在 获得 有效的 merge_ranks 之后
        if self._merge_ranks:
            SPECIAL_RANK_START = max( self._merge_ranks.values() ) + 1 # special starts from MAX_MERGE_RANK + 1
        else:
            import warnings
            warnings.warn(
                f'merge_ranks is Empty. now register special tokens right after 0 - 255 bytes.\n'
                f'run valid BPE train process to build merge_ranks forrest before register special tokens')
            SPECIAL_RANK_START = 256 # special starts from 255 + 1

        for i, sp_mark in enumerate(self._special_marks):
            self.special_tokens[sp_mark] = SPECIAL_RANK_START + i
        
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()} # special token id --> special mark str



    def _prepare_train(self, num_merges:int|None = None, *args, **kwargs):
        '''
        num_merges 是希望 tokenizer 最后达到的 merge_ranks 的大小. 它与 explicit_n_vocab(如果有)的关系是:
        explicit_n_vocab = 256 + num_merges + num_special_marks. 如果冲突, 以 explicit_n_vocab 为准

        真正在这里为训练轮次准备的 变量是 num_train_epochs
        '''
        # if explicit_n_vocab not exist, must input num_merges here
        if not hasattr(self, 'explicit_n_vocab'):
            assert isinstance(num_merges, int) and num_merges >= 0, f'num merges not set. must input num_merges >= 0.'
            self._num_merges = num_merges

        # if explicit_n_vocab already set, and num_merges input here. warning if not match
        elif isinstance(num_merges, int) and self.explicit_n_vocab - 256 - len(self._special_marks) != num_merges:
            # warning. used pre-set _num_merges
            import warnings
            warnings.warn(
                f'input `num_merges`{num_merges} is not consistent with `explicit_n_vocab` from initialization.\n'
                f'merge times derived from `explicit_n_vocab` shall be `explicit_n_vocab`-num_specials-256 = '
                f'{self.explicit_n_vocab-256-len(self._special_marks)}.\n'
                f'ignore `num_merges`, use `explicit_n_vocab` to run BPE.')
            self._num_merges = self.explicit_n_vocab-256-len(self._special_marks)

        # if explicit_n_vocab already set, and match with num_merges or num_merges is None
        else:
            self._num_merges = self.explicit_n_vocab-256-len(self._special_marks)
        
        assert self._num_merges > len(self._merge_ranks)-256, f'current size of merge_ranks - 256 must be smaller to num_merges'

        self._num_train_epochs = self._num_merges - len(self._merge_ranks)
        


    def _update_tokenizer(self, occur_most_pair:tuple[int, int], new_token:int, occurence:int|None):
        # rank = i: 0, ..., _num_mergs-1
        self._merge_ranks[occur_most_pair] = new_token
        self._vocab[new_token] = self._vocab[occur_most_pair[0]] + self._vocab[occur_most_pair[1]]

        if occurence: # occurence 不会是0
            print(f'merge process {len(self._merge_ranks)}/{self._num_merges}: {occur_most_pair} -> {new_token}'
                  f'[{self._vocab[new_token]}] had {occurence} occurences')


    def _clear(self):
        self._merge_ranks: dict[tuple[int, int], int] = {} # 初始化 _merge_ranks
        self._vocab: dict[int, bytes] = {i:bytes([i]) for i in range(256)} # 初始化 _vocab


    def bpe_single_merge(self, rank:int, tokens_generator:t.Generator, verbose:bool=False):
        # rank 从 0 开始
        agg_p_counts:t.Dict[tuple[int, int], int] = {}
        stored_tokens = []
        for tokens in tokens_generator:
            # 对于最多只有 1个token 的 tokens(即all merged together)
            # 它从本次 merge开始，不会再贡献影响任何后续 p_counts, 也没有更新的必要
            if len(tokens) <= 1:
                continue
            get_pair_counts(tokens, agg_p_counts)
            stored_tokens.append( tokens )
        
        if not agg_p_counts:
            self.save(os.path.join(self._buffer_dir, f'cache_{self.name}.tok')) # 在raise error前先保存已经train好的tokenizer防止前功尽弃
            raise_run_out_corpus_error(rank, len(self._special_marks))
        
        occur_most_pair: tuple[int, int] = max(agg_p_counts, key=agg_p_counts.get)
        occurence: int|None = agg_p_counts[occur_most_pair] if verbose else None
        new_token: int = rank + 256
        del agg_p_counts

        yield occur_most_pair, occurence, new_token # first yield

        # yield remain as new tokens_generator
        for tokens in stored_tokens:
            yield merge_pair(tokens, occur_most_pair, new_token)
    

    def train_bpe(self, corpora:str, num_merges:int|None = None, verbose:bool=False, *args, **kwargs):
        # baseTokenizer 因为没有中间结果可以缓存, 故续训（load merge_ranks 之后再输入corpus train），是没办法校对的
        # 所以 baseTokenizer 只能从头开始 train
        self._clear() # 清空 _merge_ranks & _vocab 即都置空
        self._prepare_train(num_merges) # bpe循环次数还是由 explicit_n_vocab 确定(如果存在). 若不存在, 则由输入的 num_merges 确定

        chunks_str: t.List[str] = re.findall(self.pat_str, corpora) # pre-split to list of string
        with ThreadPoolExecutor(max_workers=8) as e:
             yield_tokens = e.map(encode_to_ints, chunks_str) # a generator
        
        # <merge循环有前后依赖，所以不能并行>
        for i in range(self._num_merges): # _num_merges == _num_train_epochs, i 从 0 到 num_merges-1
            # rank := i + init_merge_ranks_size = i + 0 = i: 0, ..., _num_mergs-1
            yield_output = self.bpe_single_merge(i, yield_tokens, verbose)
            occur_most_pair, occurence, new_token = next(yield_output) # first yield
            
            self._update_tokenizer(occur_most_pair, new_token, occurence)
            
            yield_tokens = yield_output # continue to yield tokens. update yield_tokens

        # set down others
        self.explicit_n_vocab = 256 + len(self._merge_ranks) + len(self._special_marks)
        self._register_special_tokens()



    def _encode_chunk(self, tokens:t.List[int]) -> t.List[int]:
        '''
        对 chunk(tokens) 作持续的 merge, 直至 无可merge
        '''
        while len(tokens) > 1: # 在循环体内不断更新 tokens
            p_counts: t.Dict[tuple[int, int], int] = get_pair_counts(tokens)
            # 取出 合并 rank 最小的 pair. 当无可合并时, 即 pairs 中无一存在于 _merge_ranks, 那么 min_rank_pair 不存在于 _merge_ranks, 该跳出合并循环
            min_rank_pair: tuple[int, int] = min(p_counts, key=lambda p: self._merge_ranks.get(p, float('inf')))
            if min_rank_pair not in self._merge_ranks:
                break
            # 更新 tokens
            tokens = merge_pair(tokens, min_rank_pair, self._merge_ranks[min_rank_pair])
        
        return tokens

    
    def encode_ordinary(self, text:str) -> t.List[int]:
        '''
        encoding text that ignores any special tokens
        无视任何 special tokens / marks, 纯粹从 utf-8 编码后的字节流 作持续不断的 merge, 以 tokenize
        '''
        # encode 方法必须先检查有无 special_tokens. special_tokens 可以空, 也可以不在 encode 中用到.
        assert hasattr(self, 'special_tokens') # 有 / 无 special_tokens 属性, 可以区分 tokenizer 的 merge_ranks-empty / merge_ranks not-generated
        # raw
        chunks_str: t.List[str] = re.findall(self.pat_str, text) # pre-split to list of string
        # initial tokens: 可多线程加速
        chunks_tokens: t.List[list[int]] = [list( chunk.encode('utf-8') ) for chunk in chunks_str] # list of int(0..255)_list

        encoded_output: t.List[int] = []
        for tokens in chunks_tokens: # tokens: list of 0-255(int)
            encoded_output.extend( self._encode_chunk(tokens) )
        
        return encoded_output


    def encode_special(
            self,
            text:str,
            allowed_special:t.Literal["all"] | t.AbstractSet[str]
            ) -> t.List[int]:
        '''
        encoding text that first mapping registered and allowed special marks to special_token_ID, then tokenize the rest
        输入的 allowed_special 和 注册的 special_tokens 取交集.
            交集内的 special 若出现在 text 中, 则mapping as 注册的 special token ID
            text 其他部分采用 encode_ordinary 方式 编码
        '''
        # 要求本 tokenizer 已完成 specials 注册. special_tokens is dict of {str: int}
        assert hasattr(self, 'special_tokens') # 有 / 无 special_tokens 属性, 可以区分 tokenizer 的 merge_ranks-empty / merge_ranks not-generated
        if allowed_special == 'all':
            specials = self.special_tokens
        else:
            # allowed_special 交集 registered_special  如果 allowed_special 为空, 那么 specials 也为 空
            specials = {k:v for k, v in self.special_tokens.items() if k in allowed_special} # dict of str: int

        if not specials: # 如果 specials 为 空
            return self.encode_ordinary(text)
        
        special_pat = '(' + '|'.join(re.escape(k) for k in specials) + ')'
        chunks = re.split(special_pat, text) # special tokens 从 text 中分离出来: list of str

        tokens = []
        for chunk in chunks:
            if chunk in specials: # 如果是 special mark, 直接匹配 注册的 special token ID
                tokens.append( self.special_tokens[chunk] )
            else: # 如果是 plain text, 使用 encode ordinary 编码成 tokens
                tokens.extend( self.encode_ordinary(chunk) )
        
        return tokens
    
    
    def encode(self, text:str,
               allowed_special:t.Literal["all"] | t.AbstractSet[str] = set(),
               disallowed_special:t.Literal["all"] | t.Collection[str] = "all"
               ) -> t.List[int]:
        '''
        按理来说, special tokens 是拿来控制 LLM 的, 不应该出现在 text 中。这里 额外处理special的逻辑与 OpenAI tiktoken 保持一致。
        即：如果在 text 中检测到 special tokens, 则 raise error。
        
        通过参数 allowed_special/disallowed_special 来 控制 special tokens 的粒度。allow 和 disallow 的区别在于是否 raise error.

        第一步 确定 disallowed specials, 以此 判断本次 encode 要不要 raise error: 若 text 中出现了 disallowed specials, 则 raise error; 否则进入第二步
        第二步 用 encode_special 方法来 encode text: 即 allowed specials 和 registered specials 的交集会被 map to special token ID, 其余全部正常encode
        
        1. 确定 disallowed specials。若 text 里包含 disallowed specials, 则 raise error。不包含则进入下一步
            如何确定 disallowed specials?
                1. input arg disallowed_special = all: 意味着 该tokenizer 注册的 special tokens 减去 arg allowed_special, 就是 disallowed specials
                (此时若 arg allowed_special = all, 则 disallowed_special 为 空，即 没有 disallow 的 special.)
                2. input arg disallowed_special = (): 意味着 disallowed_special 为 空，即 没有 disallow 的 special.
                3. input arg disallowed_special = set of str marks: 意味着 disallowed_special 是一个 valid 集合, 检测该集合的 marks 是否出现即可.
        
        2. 若在 第1步没有 raise error, 则采用 encode with special on text. 参数 allowed_special 确定了 map to special token ID 的 special marks范围.
           这里 allowed_special 即使和上文确定的 disallowed_special 有交集也无所谓的, 因为已经保证了 text 中不存在 disallowed_special.
        '''
        assert hasattr(self, 'special_tokens') # 有 / 无 special_tokens 属性, 可以区分 tokenizer 的 merge_ranks-empty / merge_ranks not-generated

        if allowed_special == "all":
            allowed_special = set( self.special_tokens ) # dict of {str: int} ---> set of str

        if disallowed_special == "all":
            disallowed_special = set( self.special_tokens ) - allowed_special # set - set ---> set of str

        if disallowed_special: # 如果到这里, disallowed_special 非空, 那么要对 text 作检测，保证其不出现 disallowed_special, 不然 raise error
            if not isinstance(disallowed_special, frozenset):
                disallowed_special = frozenset(disallowed_special) # set --> frozenset

            if match := _special_token_regex(disallowed_special).search(text):
                raise_disallowed_special_token(match.group())
        
        return self.encode_special(text, allowed_special)

    
    def decode(self, tokens:t.List[int], errors: str = "replace") -> str:
        assert hasattr(self, '_vocab') # special_tokens / invers_special_tokens 会随着 _vocab build 而生成

        parts = []
        for idx in tokens:
            if idx in self._vocab:
                parts.append( self._vocab[idx] ) # append bytes
            elif idx in self.inverse_special_tokens:
                parts.append( self.inverse_special_tokens[idx].encode('utf-8') ) # append bytes
            else:
                raise ValueError(f'invalid index {idx} out-of-vocab')
        concat_bytes = b''.join( parts )
        # 容错：错误的字节序列使用一个 replacement char 来替代
        return concat_bytes.decode('utf-8', errors=errors)
    

    def save(self, f_path):
        '''
        保存 name
        保存 pat_str
        保存 special_marks
        保存 merge_ranks (只需要保存 keys 即可，因为 values 是从 256 的递增序列)
        ---> 即保存了一个 tokenizer 的全部信息
        '''
        assert f_path.endswith(".tok")
        with open(f_path, 'w') as f:
            # write the name of the tokenizer as version
            f.write(f"{self.name}\n")
            # write the split pattern
            f.write(f"{self.pat_str}\n")
            # write the special_marks: first line number of special marks, then each line with a special mark
            f.write(f"{len(self._special_marks)}\n")
            for mark in self._special_marks:
                f.write(f"{mark}\n")
            # write the merge_ranks' keys
            for L, R in self._merge_ranks:
                f.write(f"{L} {R}\n")

    
    def load(self, f_path):
        '''
        读取 name: line 1
        读取 pat_str: line 2
        读取 num_of_special_marks: line 3
        读取接下来 num_of_special_marks 行 --> special marks
        按序读取剩下所有行: pair tokens, 依次存入 merge_ranks[pair tokens] = 256 ++
        构建 register special tokens / vocab
        '''
        assert f_path.endswith(".tok")
        # read .tok file
        self._special_marks = []
        self._merge_ranks = {}
        with open(f_path, 'r', encoding='utf-8') as f:
            # line 1: the name of the tokenizer as version
            self.name = f.readline().strip()
            # line 2: the split pattern
            self.pat_str = f.readline().strip()
            # line 3: the num_special
            num_special = int(f.readline().strip())
            # at line 4, next num_special lines
            for _ in range(num_special):
                self._special_marks.append( f.readline().strip() )
            # all remained lines as pair merged, if exists: split and store them in order
            for i, line in enumerate(f):
                L, R = map(int, line.split())
                self._merge_ranks[(L, R)] = 256 + i # i 从 0 开始, rank 从 256 开始
        
        # 构建 vocab: token_ID --> bytes
        self._build_vocab()

        # 注册 special_tokens
        self._register_special_tokens()

    
    def view(self, tmpsave_dir):
        # _vocab: int(0 至 MAX_MERGE_RANK) --> bytes
        # _merge_ranks: (int, int) --> merged_int(256 至 MAX_MERGE_RANK)
        # special_tokens: (str, int)
        assert hasattr(self, "_vocab") and hasattr(self, "_merge_ranks") and hasattr(self, "special_tokens")
        reverse_merge = {v:k for k, v in self._merge_ranks.items()} # merged_int --> (int, int)

        from .text_preprocess import render_bytes
        import os

        with open(os.path.join(tmpsave_dir, f'tmp_{self.name}.vocab'), 'w', encoding='utf-8') as f:
            # 首先打印 special marks:
            for mark, idx in self.special_tokens.items():
                f.write(f"[{mark}] {idx}\n")

            for idx, token in self._vocab.items():
                s = render_bytes(token)
                if idx < 256:
                    f.write(f"[{s}] {idx}\n")
                else:
                    L, R = reverse_merge[idx]
                    s_L, s_R = render_bytes(self._vocab[L]), render_bytes(self._vocab[R])
                    f.write(f"[{s_L}] {L} , [{s_R}] {R} -> [{s}] {idx}\n")


    @property
    def vocab_size(self) -> int:
        return self.explicit_n_vocab
    

    @property
    def eot_token(self) -> int:
        return self.special_tokens[ENDOFTEXT]


    @functools.cached_property
    def special_marks_set(self) -> set[str]:
        return set(self.special_tokens.keys())
    

    def is_special_token(self, token: int) -> bool:
        assert isinstance(token, int)
        return token in self.inverse_special_tokens
    












import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
from collections import defaultdict
from multiprocessing import get_context, Manager
from src.utils.file_utils.remove import clean_folder
from src.common.stream_control import stream_parallel_process_with_pending

# count_pair_batch 和 merge_pair_batch 分别是完成 统计batch的token-pair occurrence信息/合并batch的token-pair到new_token 的核心函数.
# 接口设计:
# count_pair_batch: 
#   输入一个 python 结构体 tokens_offsets_border, 其中 index_0 是 tuple(np array of tokens_flat, np array of offsets), index_1 是 batch_order
#   输出一个 python 结构体,其中 index_0 是 tuple(np array of left_token, np array of right_token, np array of counts), index_1 是 batch_order

# merge_pair_batch:
#   输入一个 python 结构体 tokens_offsets_border, 其中 index_0 是 tuple(np array of tokens_flat, np array of offsets), index_1 是 batch_order
#   输入 pair_L as left_token, pair_R as right_token, new_token as merged_token of left+right
#   输出一个 python 结构体,其中 index_0 是 tuple(np array of tokens_flat, np array of offsets), index_1 是 batch_order

# count_pair_batch 和 merge_pair_batch 都是没有副作用、没有竞态风险(不存在对共享状态的修改) 的纯计算逻辑, 多进程/多线程的对比里, 多线程应该是更好的选择.
# 但是多线程的 count_pair_batch/merge_pair_batch 必须做到 release GIL, 不然无法得到有效加速. 不管是 Cython-extend, 还是 numpy, 都需要 no-gil 处理.
# 故本项目暂时全面使用 多进程处理: 虽然多了一次 data 从主进程 IPC 到 进程池 的通信成本, 但是主逻辑的编写清晰很多(无论是cython/numpy).
# 要注意: 无论是 count_pair_batch 还是 merge_pair_batch 的 result 落盘, 都直接在工作进程里完成, 不要把 result IPC回到主进程再 落盘减少一次 IPC 成本.



def count_pair_batch(tokens_offsets):
    '''
    对一个 batch 统计 pair-counts: 返回一个shape为(N, 3)的np.ndarray for pair-counts.
    3列分别是 L, R, counts. 其中 L, R 作为pair, dtype是uint16 确定. counts dtype uint64

    input:
        tokens_offsets: tokens_flat: uint16, offsets: int64

    return:
        pcounts:tuple of 3 arrays (L,R,counts), dtype(uint16, uint16, uint64)
    '''
    (tokens_flat, offsets) = tokens_offsets

    mask = np.full(shape=(len(tokens_flat),), fill_value=True)
    chunk_ends_ = (offsets-1)[1:] # 每个chunk末尾token在flat中的index
    chunk_starts_ = offsets[:-1] # 每个chunk开头token在flat中的index
    # ends_ == starts_ 的，说明chunk长度为1, 不需要统计paircounts. 略过
    # 把这些 id 指向的位置设为 False
    _where_equal_ = chunk_ends_ == chunk_starts_
    mask[ chunk_ends_[_where_equal_] ] = False
    
    mask_cp = mask.copy()

    # 提取所有非chunk end的tokens
    mask[chunk_ends_] = False
    L_tokens_flat = tokens_flat[mask] # 保持dtype=uint16, 可以为空
    
    # 提取所有非chunk start的tokens
    mask_cp[chunk_starts_] = False
    R_tokens_flat = tokens_flat[mask_cp] # 保持dtype=uint16, 可以为空

    # 构建L_tokens_flat, R_tokens_flat作为列构建的(N, 2)的pairs 2darray
    pairs = np.stack([L_tokens_flat.astype(np.uint16), R_tokens_flat.astype(np.uint16)], axis=1) # 可以为空

    # 构建新的dtype, 使得每一行L_token,R_token作为一个元素
    pair_dtype = np.dtype([('L', np.uint16), ('R', np.uint16)])
    structured = pairs.view(pair_dtype).squeeze()

    # 聚合计算频数
    uniq_pairs, counts = np.unique(structured, return_counts=True)

    # (N, 3)array as (L, R, counts), dtype分别是(uint16, uint16, uint64)
    pcounts = (uniq_pairs['L'], uniq_pairs['R'], counts.astype(np.uint64))

    return pcounts # pcounts:tuple of 3 arrays (L,R,counts), dtype(uint16, uint16, uint64)



# merge_pair 如果是一个 C/C++ 封装的接口，那么传参 tokens 时, 会遍历整个list并拷贝。
# 解决办法是把 token data 用numpy.array.data 的方式传入. 这种方式只传数组指针, 不拷贝.
# 为了统一接口, 在这里先提供一份 tokens_batch（np.ndarray）以及其他输入输出的版本
def merge_pair_batch_memcontiguous(
        tokens_offsets: object,
        pair_L:np.uint16,
        pair_R:np.uint16,
        new_token:np.uint16):
    # tokens_flat:np.ndarray of uint16
    # offsets:np.ndarray of int64
    # e.g, tokens_flat: [1, 2, 3, 4, 5], offsets: [0, 1, 1, 3, 5] --> [1], [], [2, 3], [4,5]
    # tokens_lens: [1, 0, 2, 2]
    (tokens_flat, offsets) = tokens_offsets
    tokens_lens = [j-i for i, j in zip(offsets, offsets[1:])]

    output_tokens_lens = list(tokens_lens) # 复制 tokens_lens
    output_tokens_flat = np.zeros_like(tokens_flat, dtype=np.uint16) # output tokens flat 只会比 token flat 短

    num_merges = 0
    for i in range( len(tokens_lens) ): # len(offsets)-1 == len(tokens_lens)
        # 从 tokens_flat 中slice出 tokens
        tokens = tokens_flat[offsets[i]:offsets[i+1]]
        # 遍历 tokens, 看里面是否出现了 pair0 pair1
        len_tokens, j = tokens_lens[i], 0
        while j < len_tokens:
            if j < len_tokens-1 and tokens[j] == pair_L and tokens[j+1] == pair_R:
                output_tokens_lens[i] -= 1 # 如果出现pair, 该tokens要发生一次merge, 长度-1
                output_tokens_flat[offsets[i]+j-num_merges] = new_token
                j += 2
                num_merges += 1
            else:
                output_tokens_flat[offsets[i]+j-num_merges] = tokens[j]
                j += 1
    
    output_offsets = np.array([0]+output_tokens_lens, dtype=np.int64).cumsum()

    return (output_tokens_flat[:output_offsets[-1]], output_offsets)





def merge_pair_batch_parallel(
        tokens_offsets: object,
        pair_L:np.uint16,
        pair_R:np.uint16,
        new_token:np.uint16):
    # tokens_flat:np.ndarray, # np.ndarray of uint16
    # offsets:np.ndarray, # np.ndarray of int64
    # e.g, tokens_flat: [1, 2, 3, 4, 5], offsets: [0, 1, 1, 3, 5] --> [1], [], [2, 3], [4,5]
    # tokens_lens: [1, 0, 2, 2]
    (tokens_flat, offsets) = tokens_offsets
    tokens_lens = [j-i for i, j in zip(offsets, offsets[1:])]

    output_tokens_lens = list(tokens_lens) # 复制 tokens_lens
    output_tokens_flat = np.zeros_like(tokens_flat, dtype=np.uint16) # output tokens flat 只会比 token flat 短
    output_filter = np.zeros_like(tokens_flat, dtype=np.bool) # 从 output_tokens_flat 中 filter 出 output 的 mask
    
    # can parallel-program to speed upon loop i: thread-secure
    for i in range( len(tokens_lens) ): # len(offsets)-1 == len(tokens_lens)
        # 从 tokens_flat 中slice出 tokens
        tokens = tokens_flat[offsets[i]:offsets[i+1]]
        # 遍历 tokens, 看里面是否出现了 pair0 pair1
        len_tokens, j = tokens_lens[i], 0
        while j < len_tokens:
            if j < len_tokens-1 and tokens[j] == pair_L and tokens[j+1] == pair_R:
                output_tokens_lens[i] -= 1 # 如果出现pair, 该tokens要发生一次merge, 长度-1
                output_tokens_flat[offsets[i]+j] = new_token
                output_filter[offsets[i]+j] = True
                j += 2
            else:
                output_tokens_flat[offsets[i]+j] = tokens[j]
                output_filter[offsets[i]+j] = True
                j += 1
    
    output_offsets = np.array([0]+output_tokens_lens, dtype=np.int64).cumsum()

    return (output_tokens_flat[output_filter], output_offsets)







def raise_continue_num_merges_conflict(num_merged, num_total_merges, continue_num_merges):
    raise ValueError(
        f'continue_num_merges plus loaded merge_ranks size must not exceed num_merges derived from `explicit_n_vocab`.\n'
        f'merge times derived from `explicit_n_vocab` shall be `explicit_n_vocab`-num_specials-256 = {num_total_merges}.\n'
        f'now loaded merge_ranks size {num_merged} + `continue_num_merges`{continue_num_merges} = '
        f'{num_merged+continue_num_merges} which exceeds {num_total_merges}.'
        )






# 总共执行 num_merges 次下述循环: step i, i = 0 <-> num_merges-1  --->  step i+1
# MAP 任务之 pair_count:
# parquet dataset as /tokens/i  ---> shard read batch ---> map to executor for MAP_PAIR_COUNT_BATCH
# ---> part_pair_count ---> write independently with metadata --> parquet dataset as /p_counts/i

# REDUCE 任务之
# --aggregate parquet dataset /p_counts/i ---> to_merge_pair, new_token

# MAP 任务之 pair_merge:
# parquet dataset as /tokens/i ---> shard read batch ---> map to executor for MAP_PAIR_MERGE_BATCH
# ---> part_merged_tokens ---> write independently with metadata --> parquet dataset as /tokens/i+1



# buffer_dir/tokens  ---> 存储诸如 0/. 1/. 等 parquet dataset. 其中文件名数字表达 num_merged in tokens
# buffer_dir/paircounts ---> 存储诸如 0/. 1/. 等 parquet dataset. 其中文件名数字表达对应 tokens 的 pair-counts

# batch_size: 单个工作线程/进程处理, 在单次批处理时的行数  --> num_batches = total_num_rows_after_chunk / batch_size
# 我们用 parquet dataset 来表达完整的 tokens 和 pari-counts data
# 因为 dataset 数据结构即可以视作独立的 parquet 文件, 又可以视作同一个metadata下的数据表 --> 提供天然的 REDUCE 途径, 写入批数据到dataset即可

# 同时, parquet dataset还提供了 并行读取 的途径: 
# 1. 对于多进程, 并行读取 --> 并行读取多个 dataset 内部的 parquet 文件, 所以要将 dataset 内各 fragment的路径, 及其对应的写入路径发送给工作进程,
#     工作进程拿到 读取路径、计算结果、将结果写入到对应写入路径
# 2. 对于多线程, 并行读取 --> 并行读取 dataset.get_fragments() 方法返回的 fragment 对象, 及其对应的写入路径(多线程下fragment对象不需要pickle, 可共享)
#     thread MAP 任务要尽量 绕开GIL: pa operations 基本都是释放 GIL 的, pair-count/merge也必须要重写成non-GIL的.
#     内部的一些 Python if-else/顺序/引用 操作, 尽管会持有 GIL, 但持有时间极短, 预计不会对性能有较大影响. 其余的Python操作不要放在工作线程里.

# 综上, batch_size 就是一个工作线程/进程在具体执行计算时的数据量, 也是一个 dataset.fragment 的行数.
# 它不能过大 --> num_workers*batch_size OOM,  也不能过小 --> batch_size导致 fragment file size 过小, 导致 num_fragments 过多, 影响性能.

# 那么, 是控制 batch_size 还是 num_batches?
# 答案: 控制 dataset 内部单个 parquet 文件大小在 256MB - 1GB, 因为 dataset并发读取, 单个pq文件大小比控制pq文件总数更重要. 后续分析过程如下:

# step1: 确定 batch_size
# avg_num_tokens_per_chunk, 英文约为5-10, 中文约为60.
# token_bytes, 小词表(大小<=65536)为2, 全尺寸词表(大小>65536)为4
# single paritial(from batch) file size: 
#     tokens = batch_size * avg_num_tokens_per_chunk * token_bytes
#     pcounts <= batch_size * vocab_size^2 * (2*token_bytes + 8)
# 由于 pcounts 的 生成完全是由客观情况决定的, 且 pcounts dataset 只涉及 并发单文件写入 和 聚合统计 过程, 不涉及并发读取, 所以无需不控制其大小.
# ----> 控制 batch_size * avg_num_tokens_per_chunk * token_bytes 在 256MB ~ 1024MB 之间. 中/英 语料 avg_num_tokens_per_chunk 分别取64/8.
# ----> for英文&小词表, batch_size=(16~64)M,   for英文&大词表, batch_size=( 8~32)M,
#       for中文&小词表, batch_size=( 2~ 8)M,   for中文&大词表, batch_size=( 1~ 4)M

# step2. 确定写入的 row_group_size: 由于 batch_size 完整地落在 valid row group size 区间内, 直接让 row_group_size = batch_size 是有利于并发的

# step3. 确定工作进程/线程数量:
# 根据 总可用内存, 和 batch_size 大小的批数据 单任务计算 所需的内存, 得到 任务并行数目
#   merge任务:
#     input:
#       tokens_flat:        num_tokens个uint16/uint32 = batch_size * avg_num_tokens_per_chunk * 2/4 bytes
#       offsets:            batch_size个int64         = batch_size * 8 bytes
#     output:
#       merged_tokens_lens: batch_size个int64         = batch_size * 8 bytes
#       filter:             num_tokens个bool          = batch_size * avg_num_tokens_per_chunk * 1 bytes
#       merged_tokens_flat: num_tokens个uint16/uint32 = batch_size * avg_num_tokens_per_chunk * 2/4 bytes
#   --> merge任务总结:
#       for英文&小词表, 56*batch_size(内存池32*batch_size) bytes; for英文&大词表, 88*batch_size(内存池48*batch_size) bytes
#       for中文&小词表, 336*batch_size(内存池200*batch_size) bytes; for中文&大词表, 592*batch_size(内存池328*batch_size) bytes
#   count任务:
#     input:
#       L_tokens:        num_tokens个uint16/uint32 = batch_size * avg_num_tokens_per_chunk * 2/4 bytes
#       R_tokens:        num_tokens个uint16/uint32 = batch_size * avg_num_tokens_per_chunk * 2/4 bytes
#     output:
#       keys:            num_tokens个uint32/uint64 = batch_size * avg_num_tokens_per_chunk * 4/8 bytes
#       L_uniqs:         num_tokens个uint32/uint64 = batch_size * avg_num_tokens_per_chunk * 2/4 bytes
#       R_uniqs:         num_tokens个uint32/uint64 = batch_size * avg_num_tokens_per_chunk * 2/4 bytes
#       counts:          num_tokens个uint64 = batch_size * avg_num_tokens_per_chunk * 8 bytes
#   --> count任务总结:
#       for英文&小词表, 160*batch_size(内存池128*batch_size) bytes; for英文&大词表, 256*batch_size(内存池192*batch_size) bytes
#       for中文&小词表, 1280*batch_size(内存池1024*batch_size) bytes; for中文&大词表, 2048*batch_size(内存池1536*batch_size) bytes
#   显然 count任务 所需的内存 远大于 merge任务. 由于内存池复用, 只需考虑 count任务.
#   根据batch_size范围, 单工作线程/进程任务计算 所需内存 如下:
#       for英文&小词表, 2.5~10GB(内存池2~8GB); for英文&大词表, 2~8GB(内存池1.5~6GB)
#       for中文&小词表, 2.5~10GB(内存池2~8GB); for中文&大词表, 2~8GB(内存池1.5~6GB)

# 总结(M=1024^2, G=1024^3):
# 小词表:
#   英文, batch_size=(16~64)M, 单工作线程/进程任务计算 所需内存 2.5~10GB(batch_size* 160), 其中内存池  2 ~8GB(batch_size* 128)
#   中文, batch_size=( 2~ 8)M, 单工作线程/进程任务计算 所需内存 2.5~10GB(batch_size*1280), 其中内存池  2 ~8GB(batch_size*1024)
# 大词表:
#   英文, batch_size=( 8~32)M, 单工作线程/进程任务计算 所需内存  2 ~ 8GB(batch_size* 256), 其中内存池 1.5~6GB(batch_size* 192)
#   中文, batch_size=( 1~ 4)M, 单工作线程/进程任务计算 所需内存  2 ~ 8GB(batch_size*2048), 其中内存池 1.5~6GB(batch_size*1536)

# step4: 根据本机总内存 memory_size / 内存水位 alpha, 动态确定 batch_size 和 num_workers


def text_to_byte_pa_table(pre_split_pat, text, tokens_schema):
    '''
    text_to_tokens_pa_table: 把 text 依照正则表达式 pre_split_pat 切分, 并将切分结果映射成 byte-values
    返回 一个单列pa.Table, 列名和 schema 由 tokens_schema 确定
    
    :param pre_split_pat: 预切分的正则表达式
    :param text: 文本
    :param tokens_schema: pa.Table的schema
    '''
    chunks_str = re.findall(pre_split_pat, text) # list of tokens(string)
    
    # list of list of integers(every list of integers as tokens)
    chunks_tokens = [encode_to_ints(chunk) for chunk in chunks_str]
    
    # 创建 pa table
    batch_table = pa.Table.from_pydict({tokens_schema[0].name: chunks_tokens}, tokens_schema)
    return batch_table


def text_corpora_to_init_tokens_pqfiles(
        corpora: t.List[str]|str,
        column: t.List[str]|str|None,
        pre_split_pat,
        save_dir,
        tokens_schema,
        batch_size) -> t.List[str]:
    '''
    text_corpora_to_init_tokens_pqfiles: 输入语料 corpora, 全部作 预切分+byte映射 之后, 存储 parquet 文件到 save_dir. 返回 byte-value parquet 目录列表
    
    :param corpora: 语料. 可以是 1. parquet文件列表  2. 语料文本
    :param column: parquet列名. 与corpora对应. 若column is None, 说明corpora是语料文本而不是路径
    :param save_dir: 存储 byte-value parquet 语料的目录
    :param tokens_schema: byte-value parquet 的 schema
    :param batch_size: 写入 byte-value parquet 的 row_group_size
    '''
    if isinstance(corpora, str):
        # 若 column not None 且 corpora 作为路径 --> 组装成list
        if column is not None and os.path.isfile(corpora) and os.path.exists(corpora) and corpora.endswith('.parquet'):
            corpora = [corpora]
            column = [column]
        # 若 column is None, corpora 作为语料文本
        else:
            import uuid
            init_tokens_path = os.path.join(save_dir, str(uuid.uuid3(uuid.NAMESPACE_DNS, corpora)))
            tokens_table = text_to_byte_pa_table(pre_split_pat, corpora, tokens_schema)
            pq.write_table(tokens_table, init_tokens_path)
            return [init_tokens_path]
    
    if isinstance(corpora, list):
        # 检查 column 是否完整输入
        assert isinstance(column, list) and len(corpora) == len(column), \
            f'corpora {corpora} not match with column {column}'
        # 检车 语料文件是否都存在
        assert all([os.path.isfile(corpus) and os.path.exists(corpus) and corpus.endswith('.parquet') for corpus in corpora]), \
            f'file-error with input copora {corpora}'
        
        init_tokens_pq_files = []
        for (corpus, text_col) in zip(corpora, column):
            # 在 buffer_dir/内生成 对应的 byte-value tokens parquet file
            init_tokens_path = os.path.join(save_dir, os.path.basename(corpus))
            with pq.ParquetWriter(init_tokens_path, tokens_schema) as writer:
                for batch in pq.ParquetFile(corpus).iter_batches(batch_size, columns=[text_col]):
                    text = '\n'.join( batch[text_col].to_pylist() )
                    tokens_table = text_to_byte_pa_table(pre_split_pat, text, tokens_schema)
                    writer.write_table(tokens_table, batch_size)

            init_tokens_pq_files.append(init_tokens_path)
        
        return init_tokens_pq_files




import multiprocessing as mp
import atexit
from multiprocessing.util import Finalize

import ext.bpeboost as bpeboost
import pyarrow.dataset as ds
from ..parquet.sharding import shard_pq_to_ds
from ext.bpeboost import process_count_u16pair_batch, process_merge_u16pair_batch



def _worker_init(block_size: int):
    # 子进程启动时, 执行 cython 包里的 initialize
    bpeboost.initialize_process(block_size)

    # 注册 进程退出时的清理程序
    Finalize(None, bpeboost.close_process, exitpriority=10)
    atexit.register(bpeboost.close_process)



class mpbufferBBPE_u16Tokenizer(baseBBPETokenizer):
    '''
    多进程运行缓冲的BBPE小词表分词器(uint16 token)
    '''
    token_dtype = pa.uint16()

    paircounts_schema = pa.schema([
        pa.field('L', token_dtype),
        pa.field('R', token_dtype),
        pa.field('counts', pa.uint64()),
        ])
    
    tokens_schema = pa.schema([
        pa.field( 'tokens', pa.large_list(pa.field('token', token_dtype)) ),
        ])


    @classmethod
    def _map_to_process_read_count_write(cls, src_tgt_paths):
        fragment_path, save_path = src_tgt_paths

        tokens_col = pq.read_table(fragment_path, schema = cls.tokens_schema).column(0)
        if tokens_col.num_chunks == 1:
            tokens_arr = tokens_col.chunk(0)
        else:
            tokens_arr = tokens_col.combine_chunks()
        
        tokens_flat = tokens_arr.values.to_numpy()
        offsets = tokens_arr.offsets.to_numpy()

        L, R, counts = process_count_u16pair_batch((tokens_flat, offsets))
        table = pa.Table.from_arrays([
            pa.array(L, type = cls.token_dtype),
            pa.array(R, type = cls.token_dtype),
            pa.array(counts, type = cls.paircounts_schema.field('counts').type),
            ], schema = cls.paircounts_schema)

        if table:
            pq.write_table(table, save_path)


    @classmethod
    def _map_to_process_read_merge_write(cls, src_tgt_paths, L, R, new_token):
        fragment_path, save_path = src_tgt_paths

        tokens_col = pq.read_table(fragment_path, schema = cls.tokens_schema).column(0)
        if tokens_col.num_chunks == 1:
            tokens_arr = tokens_col.chunk(0)
        else:
            tokens_arr = tokens_col.combine_chunks()

        tokens_flat = tokens_arr.values.to_numpy()
        offsets = tokens_arr.offsets.to_numpy()

        merged_tokens_flat, merged_offsets = process_merge_u16pair_batch((tokens_flat, offsets), L, R, new_token)
        merged_tokens = pa.ListArray.from_arrays(merged_offsets, merged_tokens_flat)

        table = pa.Table.from_arrays([
            pa.array(merged_tokens, type = cls.tokens_schema.field('tokens').type),
        ], schema = cls.tokens_schema)

        if table:
            pq.write_table(table, save_path)


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # buffered tokenzier 需要在 buffer 目录里创建 tokens & paircounts 两个文件目录, 以暂存中间步骤的结果
        self._buffer_tokens_dir = os.path.join(self._buffer_dir, 'tokens')
        os.makedirs(self._buffer_tokens_dir, exist_ok=True)

        self._buffer_paircounts_dir = os.path.join(self._buffer_dir, 'paircounts')
        os.makedirs(self._buffer_paircounts_dir, exist_ok=True)


    def _set_config(self, language=t.Literal['en', 'zh'], batch_size_level=t.Literal['min', 'medium', 'high', 'max'], memory_utilization=0.8):
        '''
        小词表
        en corpus --> batch_size=(16~64)M, mem_need = batch_size* 160, num_workers = memory_size*alpha/(batch_size* 160), mem_pool = batch_size* 128
        zh corpus --> batch_size=( 2~ 8)M, mem_need = batch_size*1280, num_workers = memory_size*alpha/(batch_size*1280), mem_pool = batch_size*1024
        为了简化运算, batch_size 只在 16/32/48/64(en corpus) 或 2/4/6/8(zh corpus) 中选择, 分别对应 level 'min'/'medium'/'high'/'max'
        '''
        # 只考虑 64GB 内存的机器
        memory_size = 64 * 1024 **3
        batch_size = 2 * 1024**2
        if batch_size_level == 'min':
            pass
        elif batch_size_level == 'medium':
            batch_size *= 2
        elif batch_size_level == 'high':
            batch_size *= 3
        elif batch_size_level == 'max':
            batch_size *= 4
        else:
            raise ValueError(f"wrong batch_size_level for {batch_size_level}. must be one of 'min'/'medium'/'high'/'max'.")
        
        if language == 'zh':
            memory_batchsize_coef = 1280 # memory_need = batch_size* 1280
            self.__pool_batch_size_coef = 1024 # mem_pool = batch_size* 1024
        elif language == 'en':
            batch_size *= 8
            memory_batchsize_coef = 160 # memory_need = batch_size* 160
            self.__pool_batch_size_coef = 128 # mem_pool = batch_size* 128
        else:
            raise ValueError(f"wrong language for {language}. must be one of 'en'/'zh'.")
        
        self._batch_size = batch_size
        self._num_workers = int( memory_size * memory_utilization / (batch_size*memory_batchsize_coef) )


    # 从多个 init_tokens(parquet格式), 切分成 batches 到 fragments of init dataset: tokens/0
    def _init_tokens_dataset(self, init_tokens_pq_files: t.List[str]):
        shard_pq_to_ds(
            pq_files = init_tokens_pq_files,
            save_dir = os.path.join(self._buffer_tokens_dir, '0'),
            shard_row_size = self._batch_size,
            write_metadat = False
        )


    def _start_tokens_ds(self, num_merges, executor):
        # 检查 num_merges 和 explicit_n_vocabs 和 merge_ranks 的size 之间的冲突. 确定 num_train_epochs
        super()._prepare_train(num_merges)

        # 确定 buffer_dir/tokens/ 不为空, 至少存在一个 dataset
        assert os.listdir(self._buffer_tokens_dir), f'empty buffer directory of tokens {self._buffer_tokens_dir}'

        # 如果 merge_ranks 的 size = 0, 那么本次 BPE 是从头开始train, 起始dataset 是 tokens/0/
        if len(self._merge_ranks) == 0:
            tokens_ds_0 = os.path.join(self._buffer_tokens_dir, '0')
            assert len(os.listdir(tokens_ds_0)) > 1, f'initial dataset tokens/0 {tokens_ds_0} empty. check.' # 确认 tokens dataset 0 不为空
            return tokens_ds_0
        
        # 根据最新的 tokens dataset 和 merge_ranks.size, 确认 起始dataset
        latest = max([int(f) for f in os.listdir(self._buffer_tokens_dir)])
        tokens_ds_latest = os.path.join(self._buffer_tokens_dir, f'{latest}')

        # 如果 merge_ranks 的 size > 0, 那么本次 BPE 是续train. merge_ranks.size == num_merged
        # 情况1: merge_ranks.size == latest dataset  -->  latest dataset 直接作为 起始dataset
        if latest == len(self._merge_ranks):
            return tokens_ds_latest
        # 情况2: merge_ranks.size == latest + 1 dataset --> 上次 top_pair 和 new_token 更新到 tokenizer 之后, 没有作 merge
        # 用 latest merge pair & merged_token, 对 tokens/latest 作一次 merge --> 作为 起始dataset
        elif latest + 1 == len(self._merge_ranks):
            (L, R), new_token = max( self._merge_ranks.items(), key=lambda item: item[1] )
            return self._next_tokens_ds(executor, tokens_ds_latest, L, R, new_token)
        # 情况3: 报错
        else:
            raise RuntimeError(
                f"current merge_ranks size {len(self._merge_ranks)} not match with latest buffered tokens dataset "
                f"{self._buffer_tokens_dir}/{latest}:\nmerge_ranks size shall be equal to latest, or latest + 1")
    

    def _next_tokens_ds(self, executor, tokens_ds, L, R, new_token):
        # 此时 pair merge 已经发生, merge_ranks 已经更新添加了 L, R -> merged_token
        next_rank = int( os.path.basename(tokens_ds) ) + 1
        assert next_rank == len(self._merge_ranks)

        next_tokens_ds = os.path.join(self._buffer_tokens_dir, f'{next_rank}')
        os.makedirs(next_tokens_ds, exist_ok = True)
        clean_folder(next_tokens_ds, method='all', keep=True)

        pq.write_metadata(self.tokens_schema, os.path.join(next_tokens_ds, "_common_metadata"))

        # 生成器: 生成 tokens_ds 中的 fragment 路径, 以及其对应的 merged tokens fragment 路径
        def src_tgt_path_gen(tokens_ds):
            for f in ds.dataset(tokens_ds, self.tokens_schema, format="parquet").get_fragments():
                yield (f.path, os.path.join(next_tokens_ds, os.path.basename(f.path)))

        stream_parallel_process_with_pending(
            executor = executor,
            data_gen = src_tgt_path_gen(tokens_ds),
            process_fn = self._map_to_process_read_merge_write,
            result_handler = None,
            max_pending = 8,
            process_args = (L, R, new_token)
        )

        return next_tokens_ds
    

    def _merge_info(self, executor, tokens_ds):
        # 此时 pair merge 尚未发生
        num_merged_epochs = int(os.path.basename(tokens_ds))
        assert num_merged_epochs == len(self._merge_ranks)

        # tokens_ds tokens/i  ---paircount---> paircounts_ds paircounts/i
        paircounts_ds = os.path.join(self._buffer_paircounts_dir, f'{num_merged_epochs}')
        os.makedirs(paircounts_ds, exist_ok = True)
        clean_folder(paircounts_ds, method='all', keep=True)

        pq.write_metadata(self.paircounts_schema, os.path.join(paircounts_ds, "_common_metadata"))

        # 生成器: 生成 tokens_ds 中的 fragment 路径, 以及其对应的 pair-counts fragment 路径
        def src_tgt_path_gen(tokens_ds):
            for f in ds.dataset(tokens_ds, format="parquet").get_fragments():
                yield (f.path, os.path.join(paircounts_ds, os.path.basename(f.path)))

        stream_parallel_process_with_pending(
            executor = executor,
            data_gen = src_tgt_path_gen(tokens_ds),
            process_fn = self._map_to_process_read_count_write,
            result_handler = None,
            max_pending = 8,
        )

        if len(os.listdir(paircounts_ds)) == 1: # 如果除了 _common_metadat 没有其它 parquet 文件写入
            self.save(os.path.join(self._buffer_dir, f'cache_{self.name}.tok'))
            raise_run_out_corpus_error(num_merged_epochs, len(self._special_marks))
        
        pcounts_concats = ds.dataset(paircounts_ds, format="parquet").to_table()
        agg_pcounts = pcounts_concats.group_by(['L', 'R']).aggregate([('counts', 'sum')])
        
        max_occurrence = pc.max(agg_pcounts['counts_sum'])

        filter_mask = pc.equal(agg_pcounts['counts_sum'], max_occurrence)
        _row = agg_pcounts.filter(filter_mask).slice(0, 1)

        return _row['L'][0], _row['R'][0], max_occurrence


    def train_bpe(self,
                  num_merges:int|None = None,                                   # global num merges for the tokenizer
                  *,
                  corpora:t.List[str]|str|None,                                 # None -> 从buffer_dir续train. str|parquet_paths -> 基于语料从头train
                  column:t.List[str]|str|None,                                  # 指明corpora(if parquet)数据表的列名
                  format:t.Literal['byte','text'],                              # 指明corpora(if parquet)数据类型 byte -> 0-255值; text -> string
                  language:t.Literal['en','zh'],                                # 语料的语言类型
                  batch_size_level:t.Literal['min','medium','high','max']='max',# batch_size的大小档次
                  memory_utilization:float=0.8,                                 # 对本机内存的占用水位
                  keep_window:int = 3,                                          # max reserved tokens_pq file in disk
                  verbose:bool = False
                  ):
        
        assert keep_window >= 0

        # 当 corpora is not None --> 作为语料从头train. 预处理语料(预切分/utf-8编码) 并 sharding生成第一个tokens dataset: tokens/0
        if corpora is not None:
            self._set_config(language, batch_size_level, memory_utilization) # 直接由输入参数确定 batch_size/num_workers/pool_batch_size_coef
            self._clear()
            if format == 'text': # 如果输入 corpora 类型是 text, 要经过 预切分 -> utf-8编码成byte-value -> 重新写入为 tokens_schema 的 parquet文件
                corpora = text_corpora_to_init_tokens_pqfiles(corpora, column, self.pat_str, self._buffer_dir, self.tokens_schema, self._batch_size)
            else: # corpora类型若为 byte, 则默认上述预切分/utf-8编码已经执行完成. 检查parquet即可
                assert all([os.path.isfile(corpus) and os.path.exists(corpus) and corpus.endswith('.parquet') for corpus in corpora])
            # shard 语料, 方便并行处理
            self._init_tokens_dataset(corpora)
        # 当 corpora is None --> 续train. 直接从 buffer_tokens_dir 中取最新的 dataset
        else:
            # 从tokens/latest 推断 batch_size_level
            latest = max([int(f) for f in os.listdir(self._buffer_tokens_dir)])
            tokens_ds_latest = ds.dataset(os.path.join(self._buffer_tokens_dir, f'{latest}'), format="parquet")
            avg_fragments_size = tokens_ds_latest.count_rows() // len(tokens_ds_latest.files) // 1024**2
            if language == 'zh':
                avg_fragments_size *= 8
            if avg_fragments_size <= 16:
                batch_size_level = 'min'
            elif avg_fragments_size <= 32:
                batch_size_level = 'medium'
            elif avg_fragments_size <= 48:
                batch_size_level = 'high'
            else:
                batch_size_level = 'max'
            self._set_config(language, batch_size_level, memory_utilization) # 确定 batch_size/num_workers/pool_batch_size_coef
            self._build_vocab() # vocab: token_ID --> bytes 的映射

        ctx = mp.get_context('spawn')
        memblock_size = self.__pool_batch_size_coef // 2 * self._batch_size  # // 2 是允许多一次内存块申请

        with ProcessPoolExecutor(
            max_workers = self._num_workers,
            mp_context = ctx,
            initializer = _worker_init,
            initargs = (memblock_size,)
        ) as executor:
            # 确定BPE起始的 dataset: _start_tokens_ds 会检测buffer目录里tokens_ds和num_merges, 返回符合条件的 start dataset
            tokens_ds_start = self._start_tokens_ds(num_merges, executor)
            
            # 确定BPE起始和终止的 rank: 起始 start, 终止 end(不含)
            start, end = len(self._merge_ranks), len(self._merge_ranks) + self._num_train_epochs
            
            curr_tokens_ds = tokens_ds_start
            for rank in range(start, end):
                print(f'merge rank {rank} / {start} to {end-1}')
                try:
                    L, R, max_occurrence = self._merge_info(executor, curr_tokens_ds)
                    # L, R, max_occurrence: pa.lib.UInt16Scalar, pa.lib.UInt16Scalar, pa.lib.UInt64Scalar
                    new_token, occurrence = rank + 256, int(max_occurrence) if verbose else None

                    self._update_tokenizer((int(L), int(R)), new_token, occurrence)
                    if rank == end - 1:
                        break
                    # cython-extension function 对 L, R, new_token 的输入要求是 np.uint16, np.uint16, np.uint16
                    curr_tokens_ds = self._next_tokens_ds(executor, curr_tokens_ds, np.uint16(L), np.uint16(R), np.uint16(new_token))
                    
                finally:
                    to_remove = rank - keep_window
                    if to_remove > 0: # 不删除 tokens/0
                        clean_folder(os.path.join(self._buffer_paircounts_dir, f'{to_remove}'), False)
                        clean_folder(os.path.join(self._buffer_tokens_dir,     f'{to_remove}'), False)

        # set down others
        self.explicit_n_vocab = 256 + len(self._merge_ranks) + len(self._special_marks)
        self._register_special_tokens()




from ext.bpeboost import thread_count_u32pair_batch, thread_merge_u32pair_batch

def _thread_init(block_size: int):
    # 子线程启动时, 执行 cython 包里的 initialize
    bpeboost.initialize_thread(block_size)




class mtbufferBBPE_u32Tokenizer(baseBBPETokenizer):
    '''
    多线程运行缓冲的BBPE大词表分词器(uint32 token)
    '''
    token_dtype = pa.uint32()

    paircounts_schema = pa.schema([
        pa.field('L', token_dtype),
        pa.field('R', token_dtype),
        pa.field('counts', pa.uint64()),
        ])
    
    tokens_schema = pa.schema([
        pa.field( 'tokens', pa.large_list(pa.field('token', token_dtype)) ),
        ])


    @classmethod
    def _map_to_thread_read_count_write(cls, src_tgt_paths):
        fragment, save_path = src_tgt_paths

        tokens_col = fragment.to_table().column(0)
        if tokens_col.num_chunks == 1:
            tokens_arr = tokens_col.chunk(0)
        else:
            tokens_arr = tokens_col.combine_chunks()
        
        tokens_flat = tokens_arr.values.to_numpy()
        offsets = tokens_arr.offsets.to_numpy()

        L, R, counts = thread_count_u32pair_batch((tokens_flat, offsets))
        table = pa.Table.from_arrays([
            pa.array(L, type = cls.token_dtype),
            pa.array(R, type = cls.token_dtype),
            pa.array(counts, type = cls.paircounts_schema.field('counts').type),
            ], schema = cls.paircounts_schema)

        if table:
            pq.write_table(table, save_path)


    @classmethod
    def _map_to_thread_read_merge_write(cls, src_tgt_paths, L, R, new_token):
        fragment, save_path = src_tgt_paths

        tokens_col = fragment.to_table().column(0)
        if tokens_col.num_chunks == 1:
            tokens_arr = tokens_col.chunk(0)
        else:
            tokens_arr = tokens_col.combine_chunks()

        tokens_flat = tokens_arr.values.to_numpy()
        offsets = tokens_arr.offsets.to_numpy()

        merged_tokens_flat, merged_offsets = thread_merge_u32pair_batch((tokens_flat, offsets), L, R, new_token)
        merged_tokens = pa.ListArray.from_arrays(merged_offsets, merged_tokens_flat)

        table = pa.Table.from_arrays([
            pa.array(merged_tokens, type = cls.tokens_schema.field('tokens').type),
        ], schema = cls.tokens_schema)

        if table:
            pq.write_table(table, save_path)


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # buffered tokenzier 需要在 buffer 目录里创建 tokens & paircounts 两个文件目录, 以暂存中间步骤的结果
        self._buffer_tokens_dir = os.path.join(self._buffer_dir, 'tokens')
        os.makedirs(self._buffer_tokens_dir, exist_ok=True)

        self._buffer_paircounts_dir = os.path.join(self._buffer_dir, 'paircounts')
        os.makedirs(self._buffer_paircounts_dir, exist_ok=True)


    def _set_config(self, language=t.Literal['en', 'zh'], batch_size_level=t.Literal['min', 'medium', 'high', 'max'], memory_utilization=0.8):
        '''
        大词表
        en corpus --> batch_size=( 8~32)M, mem_need = batch_size* 256, num_workers = memory_size*alpha/(batch_size* 256), mem_pool = batch_size* 192
        zh corpus --> batch_size=( 1~ 4)M, mem_need = batch_size*2048, num_workers = memory_size*alpha/(batch_size*2048), mem_pool = batch_size*1536
        为了简化运算, batch_size 只在 16/32/48/64(en corpus) 或 2/4/6/8(zh corpus) 中选择, 分别对应 level 'min'/'medium'/'high'/'max'
        '''
        # 只考虑 64GB 内存的机器
        memory_size = 64 * 1024 **3
        batch_size = 1024**2
        if batch_size_level == 'min':
            pass
        elif batch_size_level == 'medium':
            batch_size *= 2
        elif batch_size_level == 'high':
            batch_size *= 3
        elif batch_size_level == 'max':
            batch_size *= 4
        else:
            raise ValueError(f"wrong batch_size_level for {batch_size_level}. must be one of 'min'/'medium'/'high'/'max'.")
        
        if language == 'zh':
            memory_batchsize_coef = 2048 # memory_need = batch_size* 2048
            self.__pool_batch_size_coef = 1536 # mem_pool = batch_size* 1536
        elif language == 'en':
            batch_size *= 8
            memory_batchsize_coef = 256 # memory_need = batch_size* 256
            self.__pool_batch_size_coef = 192 # mem_pool = batch_size* 192
        else:
            raise ValueError(f"wrong language for {language}. must be one of 'en'/'zh'.")
        
        self._batch_size = batch_size
        self._num_workers = int( memory_size * memory_utilization / (batch_size*memory_batchsize_coef) )


    # 从多个 init_tokens(parquet格式), 切分成 batches 到 fragments of init dataset: tokens/0
    def _init_tokens_dataset(self, init_tokens_pq_files: t.List[str]):
        shard_pq_to_ds(
            pq_files = init_tokens_pq_files,
            save_dir = os.path.join(self._buffer_tokens_dir, '0'),
            shard_row_size = self._batch_size,
            write_metadat = False
        )


    def _start_tokens_ds(self, num_merges, executor):
        # 检查 num_merges 和 explicit_n_vocabs 和 merge_ranks 的size 之间的冲突. 确定 num_train_epochs
        super()._prepare_train(num_merges)

        # 确定 buffer_dir/tokens/ 不为空, 至少存在一个 dataset
        assert os.listdir(self._buffer_tokens_dir), f'empty buffer directory of tokens {self._buffer_tokens_dir}'

        # 如果 merge_ranks 的 size = 0, 那么本次 BPE 是从头开始train, 起始dataset 是 tokens/0/
        if len(self._merge_ranks) == 0:
            tokens_ds_0 = os.path.join(self._buffer_tokens_dir, '0')
            assert len(os.listdir(tokens_ds_0)) > 1, f'initial dataset tokens/0 {tokens_ds_0} empty. check.' # 确认 tokens dataset 0 不为空
            return tokens_ds_0
        
        # 根据最新的 tokens dataset 和 merge_ranks.size, 确认 起始dataset
        latest = max([int(f) for f in os.listdir(self._buffer_tokens_dir)])
        tokens_ds_latest = os.path.join(self._buffer_tokens_dir, f'{latest}')

        # 如果 merge_ranks 的 size > 0, 那么本次 BPE 是续train. merge_ranks.size == num_merged
        # 情况1: merge_ranks.size == latest dataset  -->  latest dataset 直接作为 起始dataset
        if latest == len(self._merge_ranks):
            return tokens_ds_latest
        # 情况2: merge_ranks.size == latest + 1 dataset --> 上次 top_pair 和 new_token 更新到 tokenizer 之后, 没有作 merge
        # 用 latest merge pair & merged_token, 对 tokens/latest 作一次 merge --> 作为 起始dataset
        elif latest + 1 == len(self._merge_ranks):
            (L, R), new_token = max( self._merge_ranks.items(), key=lambda item: item[1] )
            return self._next_tokens_ds(executor, tokens_ds_latest, L, R, new_token)
        # 情况3: 报错
        else:
            raise RuntimeError(
                f"current merge_ranks size {len(self._merge_ranks)} not match with latest buffered tokens dataset "
                f"{self._buffer_tokens_dir}/{latest}:\nmerge_ranks size shall be equal to latest, or latest + 1")
    

    def _next_tokens_ds(self, executor, tokens_ds, L, R, new_token):
        # 此时 pair merge 已经发生, merge_ranks 已经更新添加了 L, R -> merged_token
        next_rank = int( os.path.basename(tokens_ds) ) + 1
        assert next_rank == len(self._merge_ranks)

        next_tokens_ds = os.path.join(self._buffer_tokens_dir, f'{next_rank}')
        os.makedirs(next_tokens_ds, exist_ok = True)
        clean_folder(next_tokens_ds, method='all', keep=True)

        pq.write_metadata(self.tokens_schema, os.path.join(next_tokens_ds, "_common_metadata"))

        # 生成器: 生成 tokens_ds 中的 fragment, 以及其对应的 merged tokens fragment 路径
        def src_tgt_path_gen(tokens_ds):
            for f in ds.dataset(tokens_ds, self.tokens_schema, format="parquet").get_fragments():
                yield (f, os.path.join(next_tokens_ds, os.path.basename(f.path)))

        stream_parallel_process_with_pending(
            executor = executor,
            data_gen = src_tgt_path_gen(tokens_ds),
            process_fn = self._map_to_thread_read_merge_write,
            result_handler = None,
            max_pending = 16,
            process_args = (L, R, new_token)
        )

        return next_tokens_ds
    

    def _merge_info(self, executor, tokens_ds):
        # 此时 pair merge 尚未发生
        num_merged_epochs = int(os.path.basename(tokens_ds))
        assert num_merged_epochs == len(self._merge_ranks)

        # tokens_ds tokens/i  ---paircount---> paircounts_ds paircounts/i
        paircounts_ds = os.path.join(self._buffer_paircounts_dir, f'{num_merged_epochs}')
        os.makedirs(paircounts_ds, exist_ok = True)
        clean_folder(paircounts_ds, method='all', keep=True)

        pq.write_metadata(self.paircounts_schema, os.path.join(paircounts_ds, "_common_metadata"))

        # 生成器: 生成 tokens_ds 中的 fragment, 以及其对应的 pair-counts fragment 路径
        def src_tgt_path_gen(tokens_ds):
            for f in ds.dataset(tokens_ds, format="parquet").get_fragments():
                yield (f, os.path.join(paircounts_ds, os.path.basename(f.path)))

        stream_parallel_process_with_pending(
            executor = executor,
            data_gen = src_tgt_path_gen(tokens_ds),
            process_fn = self._map_to_thread_read_count_write,
            result_handler = None,
            max_pending = 16,
        )

        if len(os.listdir(paircounts_ds)) == 1: # 如果除了 _common_metadat 没有其它 parquet 文件写入
            self.save(os.path.join(self._buffer_dir, f'cache_{self.name}.tok'))
            raise_run_out_corpus_error(num_merged_epochs, len(self._special_marks))
        
        pcounts_concats = ds.dataset(paircounts_ds, format="parquet").to_table()
        agg_pcounts = pcounts_concats.group_by(['L', 'R']).aggregate([('counts', 'sum')])
        
        max_occurrence = pc.max(agg_pcounts['counts_sum'])

        filter_mask = pc.equal(agg_pcounts['counts_sum'], max_occurrence)
        _row = agg_pcounts.filter(filter_mask).slice(0, 1)

        return _row['L'][0], _row['R'][0], max_occurrence


    def train_bpe(self,
                  num_merges:int|None = None,                                   # global num merges for the tokenizer
                  *,
                  corpora:t.List[str]|str|None,                                 # None -> 从buffer_dir续train. str|parquet_paths -> 基于语料从头train
                  column:t.List[str]|str|None,                                  # 指明corpora(if parquet)数据表的列名
                  format:t.Literal['byte','text'],                              # 指明corpora(if parquet)数据类型 byte -> 0-255值; text -> string
                  language:t.Literal['en','zh'],                                # 语料的语言类型
                  batch_size_level:t.Literal['min','medium','high','max']='max',# batch_size的大小档次
                  memory_utilization:float=0.9,                                 # 对本机内存的占用水位
                  keep_window:int = 3,                                          # max reserved tokens_pq file in disk
                  verbose:bool = False
                  ):
        
        assert keep_window >= 0

        # 当 corpora is not None --> 作为语料从头train. 预处理语料(预切分/utf-8编码) 并 sharding生成第一个tokens dataset: tokens/0
        if corpora is not None:
            self._set_config(language, batch_size_level, memory_utilization) # 直接由输入参数确定 batch_size/num_workers/pool_batch_size_coef
            self._clear()
            if format == 'text': # 如果输入 corpora 类型是 text, 要经过 预切分 -> utf-8编码成byte-value -> 重新写入为 tokens_schema 的 parquet文件
                corpora = text_corpora_to_init_tokens_pqfiles(corpora, column, self.pat_str, self._buffer_dir, self.tokens_schema, self._batch_size)
            else: # corpora类型若为 byte, 则默认上述预切分/utf-8编码已经执行完成. 检查parquet即可
                assert all([os.path.isfile(corpus) and os.path.exists(corpus) and corpus.endswith('.parquet') for corpus in corpora])
            # shard 语料, 方便并行处理
            self._init_tokens_dataset(corpora)
        # 当 corpora is None --> 续train. 直接从 buffer_tokens_dir 中取最新的 dataset
        else:
            # 从tokens/latest 推断 batch_size_level
            latest = max([int(f) for f in os.listdir(self._buffer_tokens_dir)])
            tokens_ds_latest = ds.dataset(os.path.join(self._buffer_tokens_dir, f'{latest}'), format="parquet")
            avg_fragments_size = tokens_ds_latest.count_rows() // len(tokens_ds_latest.files) // 1024**2
            if language == 'zh':
                avg_fragments_size *= 8
            if avg_fragments_size <= 16:
                batch_size_level = 'min'
            elif avg_fragments_size <= 32:
                batch_size_level = 'medium'
            elif avg_fragments_size <= 48:
                batch_size_level = 'high'
            else:
                batch_size_level = 'max'
            self._set_config(language, batch_size_level, memory_utilization) # 确定 batch_size/num_workers/pool_batch_size_coef
            self._build_vocab() # vocab: token_ID --> bytes 的映射

        memblock_size = self.__pool_batch_size_coef // 2 * self._batch_size  # // 2 是允许多一次内存块申请

        with ThreadPoolExecutor(
            max_workers = self._num_workers,
            initializer = _thread_init,
            initargs = (memblock_size,)
        ) as executor:
            # 确定BPE起始的 dataset: _start_tokens_ds 会检测buffer目录里tokens_ds和num_merges, 返回符合条件的 start dataset
            tokens_ds_start = self._start_tokens_ds(num_merges, executor)
            
            # 确定BPE起始和终止的 rank: 起始 start, 终止 end(不含)
            start, end = len(self._merge_ranks), len(self._merge_ranks) + self._num_train_epochs
            
            curr_tokens_ds = tokens_ds_start
            for rank in range(start, end):
                print(f'merge rank {rank} / {start} to {end-1}')
                try:
                    L, R, max_occurrence = self._merge_info(executor, curr_tokens_ds)
                    # L, R, max_occurrence: pa.lib.UInt32Scalar, pa.lib.UInt32Scalar, pa.lib.UInt64Scalar
                    new_token, occurrence = rank + 256, int(max_occurrence) if verbose else None

                    self._update_tokenizer((int(L), int(R)), new_token, occurrence)
                    if rank == end - 1:
                        break
                    # cython-extension function 对 L, R, new_token 的输入要求是 np.uint32, np.uint32, np.uint32
                    curr_tokens_ds = self._next_tokens_ds(executor, curr_tokens_ds, np.uint32(L), np.uint32(R), np.uint32(new_token))
                    
                finally:
                    to_remove = rank - keep_window
                    if to_remove > 0: # 不删除 tokens/0
                        clean_folder(os.path.join(self._buffer_paircounts_dir, f'{to_remove}'), False)
                        clean_folder(os.path.join(self._buffer_tokens_dir,     f'{to_remove}'), False)

        # set down others
        self.explicit_n_vocab = 256 + len(self._merge_ranks) + len(self._special_marks)
        self._register_special_tokens()




import heapq
import typing as t
from collections import defaultdict, Counter

class Word:
    def __init__(self, tokens: t.List[int] = []):
        self._tokens = tokens

    # word ----merge_(left, right)-->new_token----> changes[(emerged_tokens_pair, +1) / (vanished_tokens_pair, -1)] 
    def merge(self, left_token, right_token, merged_token, *args, **kwargs) -> t.List[tuple]:
        changes = []
        i = 0
        while True:
            if i >= len(self._tokens)-1:
                break
            if self._tokens[i] == left_token and self._tokens[i+1] == right_token:
                if i > 0:
                    changes.append( ((self._tokens[i-1], left_token),  -1) )
                    changes.append( ((self._tokens[i-1], merged_token), 1) )
                if i+1 < len(self._tokens)-1:
                    changes.append( ((right_token, self._tokens[i+2]),  -1) )
                    changes.append( ((merged_token, self._tokens[i+2]), 1) )
                self._tokens[i] = merged_token
                self._tokens.pop(i+1)
            i += 1
        
        return changes
    
    def window_pair(self):
        for l, r in zip(self._tokens[:-1], self._tokens[1:]):
            yield (l, r)

    @property
    def tokens(self) -> t.List[int]:
        return self._tokens
    
    def __len__(self):
        return len(self._tokens)
    
    def __eq__(self, value: t.List[int]):
        return self._tokens == value



class BBPETokenizer(baseBBPETokenizer):
    '''
    采用 huggingface bpe算法优化后的 Byte-level Byte-Pair-Encoding Tokenizer --> BBPETokenzier
    bbpeTokenizer 是采用C++ boost的版本
    '''
    def train_bpe(self, corpora, num_merges = None, verbose = True, bow_min_freq: int = 1, *args, **kwargs):
        self._clear() # BBPETokenizer 和 baseBBPETokenizer 一样, 只可以从头开始BPE train, 不支持中途续train. 清空 _merge_ranks & _vocab 等在初始化时得到的属性
        self._prepare_train(num_merges) # 由 explicit_n_vocab(如果存在) 确定 _num_merges. 若不存在, 则由 num_merges 确定 _num_merges
        _num_merges = self._num_merges # _num_merges == _num_train_epochs == explicit_n_vocab 或 num_merges - num_special_marks - 256

        if isinstance(corpora, str):
            corpora = [corpora]
        corpus = '\n'.join( corpora )

        # 1. 计算 word_counts, 即 词袋 bag-of-word
        chunks_str: t.List[str] = re.findall(self.pat_str, corpus)
        BoW = Counter(chunks_str)

        # 2. 初步 tokenize, 将 string 切分成 tokens
        # unique_words 和 freqs 成为 BPE 的全部基础. 根据 freqs 是否大于等于一个阈值, 可以有效控制 BoW 的总size, 将整个过程控制在单机运行
        unique_words = []
        freqs = []

        for k, v in BoW.items():
            if v >= bow_min_freq: # 如果BoW确实存在超长尾以至于要用 bow_min_freq 来缩减, 那么典型取2, 即过滤掉只出现一次的超长尾unique_word. 这些words占用大量空间但对BPE的影响权重较小.
                unique_words.append( Word(list(k.encode('utf-8'))) )
                freqs.append( v )

        # 3. 计算 pair_counts & where_to_update & max_heap. 可并行
        pair_counts = defaultdict(int) # 记录 token-pair 的实时(正确)计数
        where_to_update = defaultdict(set) # 记录 token-pair 的倒排索引. 倒排索引集合要保证不可重复, 所以用集合set
        for pos, word in enumerate(unique_words):
            for tok_pair in word.window_pair():
                pair_counts[tok_pair] += freqs[pos]
                where_to_update[tok_pair].add(pos)
        
        max_heap = []
        for tok_pair, positions in where_to_update.items():
            to_merge_node = (tok_pair, pair_counts[tok_pair], positions)
            heapq.heappush(max_heap, (-to_merge_node[1], to_merge_node)) # 最大堆(优先序列)保证 最大频数token-pair与其在words中出现的位置索引集合 处于堆顶
        where_to_update.clear()

        # pair_counts是实时更新、保持正确的, 而max_heap中的p_cnts是懒更新, positions是不更新的.
        # 前者懒更新是指从heap pop出来时, 去和pair_counts对比, 如果不正确才更新并重新推入堆中(堆自平衡重新调整), 如果正确就拿来使用; 后者不更新是因为空扫描几个位置代价很小

        # 4. 执行 merge: 不断从 max_heap 中得到 max-occured-pair, 对 unique_words 作 inplace-merge, 产出相应 changes 来 update pair_counts & max_heap
        merge_cnts = 0 # BBPETokenizer 和 baseBBPETokenizer 一样, 只可以从头开始BPE train, 不支持中途续train
        while merge_cnts < _num_merges:
            try:
                _, (tok_pair, p_counts, positions) = heapq.heappop(max_heap)
            except IndexError:
                self.save(os.path.join(self._buffer_dir, f'cache_{self.name}.tok'))
                raise_run_out_corpus_error(merge_cnts, len(self._special_marks))

            # 最大堆 p_cnts 懒更新, positions不更新
            if p_counts != pair_counts[tok_pair]:
                to_merge_node = (tok_pair, pair_counts[tok_pair], positions)
                heapq.heappush(max_heap, (-to_merge_node[1], to_merge_node))
                continue
            
            if p_counts < 1:
                break

            new_tok = 256 + merge_cnts
            l_tok, r_tok = tok_pair
            self._update_tokenizer((l_tok, r_tok), new_tok, p_counts if verbose else None)

            changes = []
            # 这里 positions 是 不会重复的 set, 所以可并行
            for pos in positions:
                word = unique_words[pos]
                _changes = word.merge(l_tok, r_tok, new_tok) # list of (pair, signal)
                changes.extend([ (change, pos) for change in _changes]) # extended elements ((pair, signal), pos)

            for (pair, signal), pos in changes:
                pair_counts[pair] += signal*freqs[pos]
                if signal > 0:
                    where_to_update[pair].add(pos)

            # flush where_to_update 到 maxHeap
            for tok_pair, positions in where_to_update.items():
                if pair_counts[tok_pair] > 0:
                    to_merge_node = (tok_pair, pair_counts[tok_pair], positions)
                    heapq.heappush(max_heap, (-to_merge_node[1], to_merge_node))
            where_to_update.clear()
            merge_cnts += 1
            
        self.explicit_n_vocab = 256 + len(self._merge_ranks) + len(self._special_marks)
        self._register_special_tokens()





from .bag_of_word import get_BoW

class bbpeTokenizer(baseBBPETokenizer):
    '''
    data structures:
    token: u32
    Word: class with vector of tokens, and method merge(in-place change vector of tokens, return changes(vector of token-pair + signal)), and method window_pair(const iterator of token-pair)
    pos/iw/word_index: size_t, index of unique_words

    object:
    BoW: hashmap{string: u64} as {word_str: freq}, useless once unique_words & freqs established ---> save on disk
    unique_words:
        BoW.keys().encode('utf-8')---Word构造--->: vector of Word, in-place change, valid-live during the BPE
    freqs:
        BoW.values() --->: vector of word_counts(u64), const, valid-live during the BPE

    pair_counts: hashmap{u64: u64} as {token_pair: p_cnts}, insert/update v, valid-live during the BPE
    positions: unordered_set of pos corresponding to every specific token-pair, const once created, valid-live during untill the token-pair merged
    
    where_to_update: hashmap{u64: unordered_set} as {token-pair: positions}---> drain ---> empty ---> update from changes ---> a lot of insert/remove
    max_heap: max priority_queue with Merge{token-pair, p_cnts, positions} order by p_cnts(max_order)---> push/pop ---> valid-live during the BPE

    design-mode:
    for objects who are valid-live untill termination, no need for rapid re-use/re-construction ---> stay in system heap memory, std::move when transfer
    ---> unique_words, freqs, pair_counts, positions, max_heap
    for objects who are with a lot of insert/remove operations, need for rapid re-use/re-construction --> built on memory pool, except vector(.clear supports memory reuse)
    ---> where_to_update, changes(vector)

    so that is:
    where_to_update: hashmap{u64: unordered_set} as {token-pair: positions}: hashmap nodes on memory pool, but value(positions) managed on system heap memory
    changes: list(u64, int, size_t) as {token-pair, +-1 signal, position}: all plain basic datatype. can be managed on memory pool, but vector on system heap memory also can do thanks to .clear()
    unique_words / freqs / pair_counts / max_heap: stay in system heap memory
    '''

    def train_bpe(self,
                  corpora:t.List[str]|str,          # schema相同的parquet文件 或 语料文本自身
                  column:str|None,                  # 如果corpora是parquet文件, column应该是列名
                  num_merges: int|None = None,      # 若在初始化时没有输入explicit_n_vocab, 则这里确定执行BPE合并的总次数. 本tokenzier没有续train的概念，所以它就是bpe-train的循环总次数
                  verbose = True,
                  bow_min_freq: int = 1,            # 词袋BoW的最小频率. 低于此频率的word被剔除以保证内存容量
                  *args, **kwargs):
        
        self._clear() # 和 baseBBPETokenizer 一样, 只可以从头开始BPE train, 不支持中途续train. 清空 _merge_ranks & _vocab 等在初始化时得到的属性
        self._prepare_train(num_merges) # 由 explicit_n_vocab(如果存在) 确定 _num_merges. 若不存在, 则由 num_merges 确定 _num_merges
        _num_merges = self._num_merges # _num_merges == _num_train_epochs == explicit_n_vocab 或 num_merges - num_special_marks - 256

        # 0.py load BoW.parquet for unique_words(largelist of u32list, 用u32list代表word/tokens, 即用u32代表token), freqs(list of u64)
        # 1.cy->cpp cython层api调用 unique_words & freqs --address--> cython --> C++ Word构造(拷贝) --> unique_words(vector of Word), freqs(vector of const u64)
        #   cpp  unique_words & freqs --window2_token遍历--> pair_counts(hashmap{u64: u64}), where_to_update(hashmap{u64: unordered_set}). 该步骤可以并行(需要pair_counts&where_to_update线程安全)
        #   cpp  移动语义遍历where_to_update: pair & move(positions) + pair_counts --> C++ Merge构造(移动) --heapify--> max_heap(8-ary heap of Merge)
        # loop.cpp 初始化一个记录合并的 merges(vector of (u64, u64), 第一个u64代表两个u32合并, 第二个u64是该pair的计数), 循环merge_cnts从0到 _num_merges
        #   2. 从max_heap取顶端Merge. 如果取顶失败, 说明pair已经全部merge完毕, throw一个runoutError.
        #      拿到max Merge{pair, p_cnts, positions}. 如果 p_cnts 与 pair_counts[pair] 对不上, 更新该 Merge.p_cnts, push该Merge回max_heap, continue循环以重新取顶
        #      取到max Merge{pair, p_cnts, positions}而且p_cnts相符. 如果p_cnts<1, 退出循环. 拿到待合并pair和p_cnts, 记录其在 merges 中, 算出new_token.
        #   3. 初始化一个线性容器changes, 遍历(可并行,但需要容器changes共享线程安全)positions中的pos, 即执行:
        #      取出unique_words[pos]位置的word, 执行merge方法(待合并pair, new_token), 产出该pos局部线程的 local_changes(list(u64,int)). 把所有local_changes安全聚合到changes, 并给每个元素标记其pos
        #   4. 线性扫描changes, 取出每个元素(pair, change, pos): 用change和freqs[pos]更新pair_counts[pair]; 为change>0的pair, 插入或更新where_to_update[pair]添加pos
        #   5. 移动语义遍历where_to_update: pair & move(positions) + pair_counts --> C++ Merge构造 --push--> max_heap
        # 6.cpp->cy loop结束或跳出. 返回 merges(vector of (u64, u64)) --> merges(vector of ((u32, u32), u64)) 到cython层; 如果是throw runoutError, 该如何返回 merges 到cython层?
        # 7.cy->py 在cython层执行把 merges(vector of ((u32, u32), u64)) 转换为 list of (u32-pair, p_cnts) 作为 bpe_core_loop 的返回. 在python层依据其一次性全部更新入tokenizer

        # 执行 BoW
        BoW_pq_path = os.path.join(self._buffer_dir, 'bow.parquet')
        get_BoW(corpora, column, self.pat_str, 'u32list', bow_save_path = BoW_pq_path, bow_save_colnames = ('word', 'freq'))

        # 读取 BoW parquet 文件, 零拷贝获取pyarrow内存地址. 将这些内存地址上的数据作为副本, 拷贝构建至 cpp的 unique_words(vector of Word), 使得 unique_words 具备独立的生命周期
        # unique_words 具备与 buffer解耦的 独立的 生命周期, 这非常重要, 因为 unique_words 要执行 in-place 数据变动(merge替换并减少token)

        # 零拷贝获取底层数据地址: 读取read_table -> 合并分块combine_chunks -> 获取buffers -> addresses
        BoW = pq.read_table(BoW_pq_path, filters = ('freq', '>=', bow_min_freq))
        
        word_arr = BoW.column('word').combine_chunks() # 关键：返回 arr 保持引用，防止在 Cython 调用期间被 GC
        word_buf = word_arr.buffers() # 底层buffers large_list类型返回: [validity, offsets, values(tokens)]
        offsets_ptr = word_buf[1].address  # int64类型地址. large_list 的 offsets类型是 int64/long long, 长度是 num_words+1
        tokens_ptr = word_buf[2].address   # uint32类型地址. large_list of u32list. flattened tokens 类型是 uint32
        num_words = len(word_arr) # int -> size_t
        
        freq_arr = BoW.column('freq').combine_chunks()
        freq_buf = freq_arr.buffers() # 底层buffers u64类型返回: [validity, values(freqs)]
        freqs_ptr = freq_buf[1].address # uint64类型地址
        # num_freqs 应该等于 num_words, 无需再取

        # 传给 cython函数 bpe_train_loop: _num_merges & tokens_ptr & offsets_ptr & freqs_ptr & num_words & word_arr & freq_arr
        # 留在cython层即可(只要传进入cython函数，就可以保证其在cython函数return前保持alive避免gc): word_arr & freq_arr
        #   传给 cpp函数 c_nonpar_bpe(非并行) / c_par_bpe(并行):
        #       _num_merges(bpe循环的最大次数)
        #       num_words(word个数 = freqs长度 = offsets长度 - 1)
        #       tokens_ptr(初始 u32 tokens的起始地址)
        #       offsets_ptr(区分tokens的int64偏移值的起始地址)
        #       freqs_ptr(word对应频率uint64 freqs的起始地址)


        #   cpp函数 c_nonpar_bpe 传出:
        #       merges(vector of(u32 token-pair: (u32, u32), token-pair counts: u64))
        # cython函数 bpe_train_loop 传出: merges(tuple/list of (u32 token-pair, p_cnts)
        merges = bpe_train_loop(_num_merges, tokens_ptr, offsets_ptr, freqs_ptr, num_words, word_arr, freq_arr)

        for i, (l_tok, r_tok), p_counts in enumerate(merges):
            new_tok = i + 256
            self._update_tokenizer((l_tok, r_tok), new_tok, p_counts if verbose else None)

        self.explicit_n_vocab = 256 + len(self._merge_ranks) + len(self._special_marks)
        self._register_special_tokens()