#encoding:utf-8
from os import path
import multiprocessing
BASE_DIR = 'pyword2vec'

configs = {
    'data_path': path.sep.join([BASE_DIR,'dataset/raw/zhihu.txt']),   # 总的数据，一般是将train和test何在一起构建语料库
    'model_save_path': path.sep.join([BASE_DIR,'output/checkpoints/word2vec.pth']),

    'vocab_path': path.sep.join([BASE_DIR,'dataset/processed/vocab.pkl']), # 语料数据
    'pytorch_embedding_path': path.sep.join([BASE_DIR,'output/embedding/pytorch_word2vec2.bin']),
    'gensim_embedding_path':path.sep.join([BASE_DIR,'output/embedding/gensim_word2vec.bin']),

    'log_dir': path.sep.join([BASE_DIR, 'output/log']),           # 模型运行日志
    'figure_dir': path.sep.join([BASE_DIR, 'output/figure']),     # 图形保存路径
    'stopword_path': path.sep.join([BASE_DIR,'dataset/stopwords.txt']),

    'vocab_size':30000000,
    'embedding_dim':100,
    'epochs':6,
    'batch_size':256,
    'window_size':5,
    'negative_sample_num':5,
    'n_gpus':[1],
    'min_freq':5,
    'sample':1e-3,

    'num_workers':multiprocessing.cpu_count(),
    'learning_rate':0.025,
    'weight_decay':5e-4,
    'lr_min':0.00001,
    'lr_patience': 3, # number of epochs with no improvement after which learning rate will be reduced.
    'mode': 'min',    # one of {min, max}
    'monitor': 'loss',  # 计算指标
}
