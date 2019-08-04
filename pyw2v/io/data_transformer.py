#encoding:utf-8
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DataTransformer(object):
    def __init__(self,
                 embedding_path):
        self.embedding_path = embedding_path
        self.reset()

    def reset(self):
        self.load_embedding()

    # 加载词向量矩阵
    def load_embedding(self, ):
        print(" load emebedding weights")
        self.embeddings_index = {}
        self.words = []
        self.vectors = []
        f = open(self.embedding_path, 'r',encoding = 'utf8')
        for line in f:
            values = line.split(' ')
            try:
                word  = values[0]
                self.words.append(word)
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
                self.vectors.append(coefs)
            except:
                print("Error on ", values[:2])
        f.close()
        self.vectors = np.vstack(self.vectors)
        print('Total %s word vectors.' % len(self.embeddings_index))

    # 计算相似度
    def get_similar_words(self, word, w_num=10):
        if word not in self.embeddings_index:
            raise ValueError('%d not in vocab')
        current_vector = self.embeddings_index[word]
        result = cosine_similarity(current_vector.reshape(1, -1), self.vectors)
        result = np.array(result).reshape(len(self.words), )
        idxs = np.argsort(result)[::-1][:w_num]
        print("<<<" * 7)
        print(word)
        for i in idxs:
            print("%s : %.3f\n" % (self.words[i], result[i]))
        print(">>>" * 7)




