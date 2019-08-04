# encoding:utf-8
import argparse
from pyw2v.common.tools import logger,init_logger,seed_everything
from pyw2v.config.basic_config import configs as config
from pyw2v.model.nn import gensim_word2vec
from pyw2v.preprocessing.preprocessor import Preprocessor


def run(args):
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
                                              size=args.embedd_dim,
                                              window=args.window_size,
                                              min_count=args.min_freq,
                                              save_path=config['gensim_embedding_path'],
                                              num_workers=args.num_workers,
                                              seed = args.seed)
    word2vec_model.train_w2v([[word for word in document] for document in examples])

def main():
    parser = argparse.ArgumentParser(description='Gensim Word2Vec model training')
    parser.add_argument("--model", type=str, default='gensim_word2vec')
    parser.add_argument("--task", type=str, default='training word vector')
    parser.add_argument('--seed', default=2018, type=int,
                        help='Seed for initializing training.')
    parser.add_argument('--resume', default=False, type=bool,
                        help='Choose whether resume checkpoint model')
    parser.add_argument('--embedd_dim', default=300, type=int)
    parser.add_argument('--spochs', default=6, type=int)
    parser.add_argument('--window_size', default=5, str=int)
    parser.add_argument('--n_gpu', default='0', type=str)
    parser.add_argument('--min_freq', default=5, type=int)
    parser.add_argument('--sample', default=1e-3, type=float)
    parser.add_argument('--negative_sample_num', default=5, type=int)
    parser.add_argument('--learning_rate', default=0.025, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--vocab_size', default=30000000, type=int)
    parser.add_argument('--num_workers',default=10)
    args = parser.parse_args()
    init_logger(log_file=config['log_dir'] / (args.model + ".log"))
    logger.info("seed is %d" % args['seed'])
    seed_everything(seed=args['seed'])
    run(args)
if __name__ == "__main__":

    main()

