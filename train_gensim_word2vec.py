# encoding:utf-8
import argparse
from pyword2vec.utils.logginger import init_logger
from pyword2vec.config.word2vec_config import configs as config
from pyword2vec.model.nn import gensim_word2vec
from pyword2vec.preprocessing.preprocessor import Preprocessor


def main():
    logger = init_logger(log_name='gensim_word2vec', log_dir=config['log_dir'])
    logger.info('load data from disk' )
    processing = Preprocessor(min_len=2,stopwords_path=config['stopword_path'])
    examples = []
    with open(config['data_path'], 'r') as fr:
        for i, line in enumerate(fr):
            # 数据首行为列名
            if i == 0 and False:
                continue
            line = line.strip("\n")
            line = processing(line)
            if line:
                examples.append(line.split())

    logger.info("initializing emnedding model")
    word2vec_model = gensim_word2vec.Word2Vec(sg = 1,
                                              iter = 10,
                                              logger=logger,
                                              size=config['embedding_dim'],
                                              window=config['window_size'],
                                              min_count=config['min_freq'],
                                              save_path=config['gensim_embedding_path'],
                                              num_workers=config['num_workers'],
                                              seed = args['seed'])

    word2vec_model.train_w2v([[word for word in document] for document in examples])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='PyTorch model training')
    ap.add_argument('-s',
                    '--seed',
                    default=2018,
                    type=str,
                    help='Seed for initializing training.')
    args = vars(ap.parse_args())
    main()