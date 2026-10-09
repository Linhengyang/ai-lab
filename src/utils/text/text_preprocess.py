import collections
import random
import math
import typing as t
import re
import unicodedata


def count_corpus(sentences: list|tuple|set) -> t.Dict:
    '''
    inputs: nested container of tokens
        sentences can be nested container with basic elements as tokens

    returns: A dictionary, denoted as D
        keys are tokens(words, chars) and values are frequencies

    explains:
        Count token frequencies 
    '''
    def flatten(lst):
        for item in lst:
            if isinstance(item, (list, tuple, set)):
                yield from flatten(item)
            else:
                yield item
    
    # flatten
    tokens = list( flatten(sentences) )
    
    return collections.Counter(tokens)



def add_space_around_puncs(
        text,
        separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        ) -> str:
    '''
    在字符串中所有标点符号前后, 如果没有空格的话, 添加一个空格。连续的标点符号之间只添加一个空格

    inputs:
        text: 输入字符串
        separate_puncs: 标点符号
    returns:
        处理后的字符串
    '''
    # 正则里的高级特性：零宽断言, 匹配位置 不消耗字符
    # (?<=exp) 匹配 前面是 exp 的位置
    text = re.sub(r"(?<=\S)" + "([" + separate_puncs + "])", r" \1", text) # 匹配 前面是 \S(非空字符) 的 标点符号（位置）, 替换成 空格+该位置\1
    text = re.sub("([" + separate_puncs + "])" + r"(?=\S)", r"\1 ", text) # 匹配 后面是 \S(非空字符) 的 标点符号（位置）, 替换成 该位置\1+空格

    return text



def preprocess_space(
        text,
        need_lower=True,
        separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
        normalize_whitespace=True
        ) -> str:
    '''
    inputs:
        text
        need_lower: Bool, default as True. If true, then input str will be lowered
        separate_puncs: punctuations that shall be seen as independent tokens, such as ,.!?
        normalize_whitespace: False -> 制表符换行符等其他种类空白字符保留, 连续空格保留; True -> 以上全部转换为 单个单空格

    returns: A str obejct
        whose spaces are normal single space ' ', and single space is inserted before every independent token. left/right space trimed

    explains:
        preprocess spaces inside a str obeject
        参数 separate_puncs 确认了 作为独立token的标点符号. 独立token前面会加单空格
        参数 normalize_whitespace 确认了如何处理 单空格之外的空白字符.
            当 它为 True 时, 所有 空白字符和连续单空格都被处理为 单个单空格（副作用：当使用单空格来分割时, 只有word和punc被视作token）
            当 它为 False时, 诸如制表符和换行符之类的空白字符, 以及连续单空格都被保留（副作用：当使用单空格来分割时, 非单空格的空白字符和空字符也被视作token）
        text的左右空白都会被trim
    '''
    # 替换不间断特殊空格为单空格, 消除零宽度空白, 并trim首尾空格
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').replace('\u2060', '').replace('\ufeff', '').strip()

    if need_lower:
        text = text.lower()
    
    # 在非空字符和 separate_puncs 之间插入空格

    # ## 判断 当前字符是否是separate_puncs, 且前一个字符不是空格

    # def no_space(char, prev_char):
    #     return char in set(separate_puncs) and prev_char != " "
    # ## 从第二个字符开始遍历。如果它是separate_puncs且前一个字符不是空格, 则将它变成 " "+标点
    # out = [ " " + char if i > 0 and no_space(char, text[i-1]) else char for i, char in enumerate(text)]
    # out_str = "".join(out)

    out_str = add_space_around_puncs(text, separate_puncs)
    
    # 如果 normalize_whitespace = True, 那么把 所有其他种类的空白字符(\t \n)以及多个连续的单空格 转换为 单个单空格 
    if normalize_whitespace:
        out_str = re.sub(r'\s+', ' ', out_str)
    
    out_str = re.sub(r' +', ' ', out_str) # 把可能出现的多个连续单空格 转换成 单个单空格
    return out_str



def subsample(sentences:t.List[t.List[str]], vocab, thr=1e-4):
    '''
    sentences 是2D list of lists of text-tokens.
    vocab是sentences的词汇字典
    
    subsample是降采样(下采样), 即对高频(频率高于thr)的token, 以 sqrt( thr /freq_rate(token) )的概率保留
    <unk>未知字符代表的是所有「超低频」字符, 所以不应该带在降采样之列. 在subsample过程中会将<unk>以0概率保留(即去除)

    return:
        subsampled sentences(word tokens) 
        counter(count every token's frequency except <unk> before subsampling)
    '''
    # exclude <unk>
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]
    # count all tokens
    counter = count_corpus(sentences)
    num_all_tokens = sum(counter.values())

    def keep(token):
        return random.uniform(0, 1) < math.sqrt(thr/ counter[token] * num_all_tokens)

    return [[token for token in line if keep(token)] for line in sentences], counter



def attach_EOW_token(
        text: str,
        eow_tok: str
        ) -> str:
    '''
    inputs:
        text: string
        eow_tok: 该 end-of-word token 将被插入到每个 单空格之前. eow_tok 可以是空字符, 这样 输入输出 text 不变

    returns:
        whose spaces are normal single space ' ', and every space has append_punc append before it

    explains:
        因为 subword 会拆分整个word, eow_token 帮助区分subword之间和word之间的分割。该 eow_token 将被插入到每个 单空格之前
    '''
    text = text.strip() + " "

    words_and_puncs = text.split(" ") # 末尾包含 一个空字符串 ""
    _SPACE = eow_tok+" "
    
    return _SPACE.join(words_and_puncs).strip() # 剪去 最后面的 空格



def text_atomize(text, reserved_combos:t.List[str]=[], uniq=False) -> t.List[str]:
    '''
    按顺序分割字符串到不可分割字符(保留 reserved_combos 里的组合不拆分, 其他拆分成单个字符)
    reserved_combos里可以有正则表达式.
    返回list
    '''
    # 如果 reserved_combos 为空, 那么就是不需要保留组合, 直接 用 set/list 返回原子化结果
    if not reserved_combos:

        if uniq: # 如果需要 unique
            return list( set(text) )
        
        return list(text)
    
    rsvd_tokn_pattern = "(" + "|".join(reserved_combos) + ")" # (<unk>|...|<eos>)

    pattern = re.compile( rsvd_tokn_pattern + "|(.)" ) # (<unk>|...|</w>)|(.)  保留字符匹配group1, 其他所有单个字符匹配group2
    result = []
    for match in re.finditer(pattern, text):
        result.append( match.group(1) if match.group(1) else match.group(2) ) # 如果 match 匹配到的部分（group(1)或group(2)）

    if uniq: # 如果需要 unique
        result = list( set(result) )
    
    return result





def replace_control_characters(text: str) -> str:
    chars = []
    for ch in text:
        if unicodedata.category(ch)[0] != 'C':
            chars.append( ch ) # ch is ok
        else:
            chars.append( f"\\u{ord(ch):04x}" ) # escape
    return "".join(chars)



def render_bytes(b: bytes) -> str:
    '''
    pretty print bytes, escaping control characters
    '''
    text = b.decode('utf-8', errors='replace')
    return replace_control_characters(text)