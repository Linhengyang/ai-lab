import yaml
import typing as t
import pyarrow.parquet as pq
import pyarrow as pa
import os
import regex as re
from src.utils.text.tokenizer import mpbufferBBPE_u16Tokenizer, mtbufferBBPE_u32Tokenizer

configs = yaml.load(open('src/apps/bpe_build/configs.yaml', 'rb'), Loader=yaml.FullLoader)

################# directories ##################


################# buffer/save in workspace/cache ##################
buffer_dir = os.path.join( configs['cache_dir'], 'bpe_build/buffer/' )
tokenizer_save_dir = os.path.join( configs['artifact_dir'], configs['app_name'], 'tokenizer' )
vocab_cache_dir = os.path.join( configs['artifact_dir'], configs['app_name'], 'vocab' )




################# backup init to data/TinyStories/bytes ##################
num_merges = configs['num_merges']
save_tok_name = configs['save_tok_name']






def bpe_train():
    print('begin to run BPE on dataset TinyStories')
    train_pq = configs['train_pq']
    valid_pq = configs['valid_pq']
    for folder in [buffer_dir, tokenizer_save_dir, vocab_cache_dir]:
        os.makedirs(folder, exist_ok=True)
        
    tok = mtbufferBBPE_u32Tokenizer(name=save_tok_name, buffer_dir=buffer_dir)
    corpora = [valid_pq, train_pq]
    column = ['text']*len(corpora)

    tok.train_bpe(num_merges = 3,
                  corpora = corpora,
                  column = column,
                  format = 'text',
                  language = 'en',
                  batch_size_level = 'medium',
                  keep_window = 0,
                  verbose = True
                  )

    # save tokenizer
    tok_fpath = os.path.join(tokenizer_save_dir, f'{tok.name}.tok')
    tok.save(tok_fpath)
    # view vocab
    tok.view(tmpsave_dir = vocab_cache_dir)
    
    print('BPE ends')





def bpe_continue(tok_path:str|None):
    print('continue to run BPE on dataset TinyStories')
    tok = mpbufferBBPE_u16Tokenizer(name='init', buffer_dir='../cache/temp/buffer')

    if tok_path and os.path.isfile(tok_path):
        tok.load(tok_path)

    # continue train 只要 buffer_dir 的环境正确, load 的tok正确, 就会自动分析该读取的中间文件，以继续训练
    # 当 tok_path 为 None 时, 相当于 load 一个只有 0-255 值映射的 tokenizer, 配合 buffer_dir/tokens/0 续train

    # continue train 下, corpora必须显式地输入None
    tok.train_bpe(num_merges = 3,
                  corpora = None,
                  column = None,
                  format = 'byte',
                  language = 'en',
                  batch_size_level = 'medium',
                  memory_utilization = 0.8,
                  keep_window = 3,
                  verbose = True
                  )

    # rename and save the updated tokenizer
    tok.name = '3merges'
    tok_fpath = os.path.join(tokenizer_save_dir, f'{tok.name}.tok')
    tok.save(tok_fpath)

    # # view vocab
    # tok.view(tmpsave_dir = vocab_cache_dir)
    # print('BPE ends')