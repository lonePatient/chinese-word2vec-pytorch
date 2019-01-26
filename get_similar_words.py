#encoding:utf-8
import os
import warnings
from pyword2vec.io.data_transformer import DataTransformer
from pyword2vec.config.word2vec_config import configs as config
# from pycw2vec.config.cw2vec_config import configs as config
warnings.filterwarnings("ignore")

def main():

    data_transformer = DataTransformer(embedding_path = config['pytorch_embedding_path'])
    data_transformer.get_similar_words(word = '中国',w_num=10)
    data_transformer.get_similar_words(word='男人', w_num=10)

    del data_transformer
    #
    # data_transformer = DataTransformer(embedding_path = config['vector_save_path'])
    # data_transformer.get_similar_words(word = '中国',w_num=10)
    # del data_transformer
    # data_transformer = DataTransformer(embedding_path=config['gensim_embedding_path'])
    # data_transformer.get_similar_words(word='中国', w_num=10)

if __name__ =="__main__":
    main()
