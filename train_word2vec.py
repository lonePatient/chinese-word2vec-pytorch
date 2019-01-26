#encoding:utf-8
import argparse
import torch
import warnings
from torch import optim
from pyword2vec.train.trainer import Trainer
from pyword2vec.io.dataset import DataLoader
from pyword2vec.model.nn.skip_gram import SkipGram
from pyword2vec.utils.logginger import init_logger
from pyword2vec.utils.utils import seed_everything
from pyword2vec.config.word2vec_config import configs as config
from pyword2vec.callback.lrscheduler import StepLr
from pyword2vec.callback.trainingmonitor import TrainingMonitor
warnings.filterwarnings("ignore")

# 主函数
def main():
    arch = 'word2vec'
    logger = init_logger(log_name=arch, log_dir=config['log_dir'])
    logger.info("seed is %d"%args['seed'])
    seed_everything(seed = args['seed'])

    #**************************** 加载数据集 ****************************
    logger.info('starting load train data from disk')
    train_dataset   = DataLoader(skip_header  = False,
                                    negative_num = config['negative_sample_num'],
                                    batch_size   = config['batch_size'],
                                    window_size  = config['window_size'],
                                    data_path    = config['data_path'],
                                    vocab_path   = config['vocab_path'],
                                    vocab_size   = config['vocab_size'],
                                    min_freq     = config['min_freq'],
                                    shuffle      = True,
                                    seed         = args['seed'],
                                    sample       = config['sample']
                                    )

    # **************************** 模型和优化器 ***********************
    logger.info("initializing model")
    model = SkipGram(embedding_dim = config['embedding_dim'],vocab_size = len(train_dataset.vocab))
    optimizer = optim.SGD(params = model.parameters(),lr = config['learning_rate'])

    # **************************** callbacks ***********************
    logger.info("initializing callbacks")
    # 监控训练过程
    train_monitor = TrainingMonitor(fig_dir  = config['figure_dir'],
                                    json_dir = config['log_dir'],
                                    arch     = arch)
    # 学习率机制
    lr_scheduler = StepLr(optimizer=optimizer,
                          init_lr  = config['learning_rate'],
                          epochs   = config['epochs'])

    # **************************** training model ***********************
    logger.info('training model....')
    trainer = Trainer(model            = model,
                      vocab            = train_dataset.vocab,
                      train_data       = train_dataset,
                      optimizer        = optimizer,
                      epochs           = config['epochs'],
                      logger           = logger,
                      training_monitor = train_monitor,
                      lr_scheduler     = lr_scheduler,
                      n_gpu            = config['n_gpus'],
                      model_save_path  = config['model_save_path'],
                      vector_save_path = config['pytorch_embedding_path']
                      )
    # 查看模型结构
    trainer.summary()
    # 拟合模型
    trainer.train()
    # 释放显存
    if len(config['n_gpus']) > 0:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='PyTorch model training')
    ap.add_argument('-s',
                    '--seed',
                    default=2018,
                    type = int,
                    help = 'Seed for initializing training.')

    ap.add_argument('-r',
                    '--resume',
                    default = False,
                    type = bool,
                    help = 'Choose whether resume checkpoint model')
    args = vars(ap.parse_args())
    main()

