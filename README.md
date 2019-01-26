word2vec implementation for skip-gram in pytorch

本repo包含了使用pytorch实现skip-gram版本的word2vec词向量模型。

### 目录结构

主要的代码目录结果如下所示:

```text
├── pyword2vec
|  └── callback
|  |  └── lrscheduler.py　　
|  └── config
|  |  └── word2vec_config.py　　
|  └── dataset　　　　　　　　　　　
|  └── io　　　　　　　　　　　　　　
|  └── model
|  └── output　　　　　　　　　　　
|  └── preprocessing　　　　
|  └── train
|  └── utils
├── get_similar_words.py
├── train_gensim_word2vec.py
├── train_word2vec.py
```
### run

1. 修改｀config｀目录中的`word2vec_config.py`的配置信息
2. 运行命令`python train_word2vec.py`

### results

大概6次epochs之后，可得到以下结果:

| 目标词  |   Top10    | 目标词  |    Top10    |
| :--: | :--------: | :--: | :---------: |
|  中国  | 中国 : 1.000 |  男人  | 男人 : 1.000  |
|  中国  | 美国 : 0.651 |  男人  | 女人 : 0.764  |
|  中国  | 日本 : 0.578 |  男人  | 女生 : 0.687  |
|  中国  | 国家 : 0.560 |  男人  | 男生 : 0.670  |
|  中国  | 发展 : 0.550 |  男人  | 喜欢 : 0.625  |
|  中国  | 文化 : 0.529 |  男人  | 恋爱 : 0.601  |
|  中国  | 朝鲜 : 0.512 |  男人  |  岁 : 0.590  |
|  中国  | 经济 : 0.504 |  男人  |  女 : 0.588  |
|  中国  | 世界 : 0.493 |  男人  | 感觉 : 0.586  |
|  中国  | 社会 : 0.481 |  男人  | 男朋友 : 0.581 |

