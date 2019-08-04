#encoding:utf-8
import os
import warnings
from pyw2v.io.data_transformer import DataTransformer
from pyw2v.config.basic_config import configs as config
warnings.filterwarnings("ignore")

def main():

    data_transformer = DataTransformer(embedding_path = config['pytorch_embedding_path'])
    data_transformer.get_similar_words(word = '中国',w_num=10)
    data_transformer.get_similar_words(word='男人', w_num=10)

    del data_transformer

if __name__ =="__main__":
    main()
