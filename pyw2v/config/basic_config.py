#encoding:utf-8
from pathlib import Path
BASE_DIR = Path('pyw2v')

configs = {
    'data_path': BASE_DIR / 'dataset/raw/zhihu.txt',
    'model_save_path': BASE_DIR / 'output/checkpoints/word2vec.pth',

    'vocab_path': BASE_DIR / 'dataset/processed/vocab.pkl', # 语料数据
    'pytorch_embedding_path': BASE_DIR / 'output/embedding/pytorch_word2vec2.bin',
    'gensim_embedding_path':BASE_DIR / 'output/embedding/gensim_word2vec.bin',

    'log_dir': BASE_DIR / 'output/log',
    'figure_dir': BASE_DIR / 'output/figure',
    'stopword_path': BASE_DIR / 'dataset/stopwords.txt'
}
