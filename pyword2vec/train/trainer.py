#encoding:utf-8
import time
import numpy as np
import torch
from tqdm import tqdm
from ..callback.progressbar import ProgressBar
from .train_utils import model_device

# 训练包装器
class Trainer(object):
    def __init__(self,model,
                 epochs,
                 logger,
                 n_gpu,
                 vocab,
                 model_save_path,
                 vector_save_path,
                 train_data,
                 optimizer,
                 lr_scheduler,
                 training_monitor,
                 verbose = 1):
        self.model            = model
        self.train_data       = train_data
        self.epochs           = epochs
        self.optimizer        = optimizer
        self.logger           = logger
        self.verbose          = verbose
        self.training_monitor = training_monitor
        self.lr_scheduler     = lr_scheduler
        self.n_gpu            = n_gpu
        self.vocab            = vocab
        self.vector_save_path = vector_save_path
        self.model_save_path  = model_save_path
        self._reset()

    def _reset(self):
        self.progressbar       = ProgressBar(loss_name='loss',n_batch=len(self.vocab)* 50 )
        self.model,self.device = model_device(n_gpu=self.n_gpu,model = self.model,logger = self.logger)
        self.start_epoch = 1

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        # 总的模型参数量
        self.logger.info('trainable parameters: {:4}M'.format(params / 1000 / 1000))
        # 模型结构
        self.logger.info(self.model)

    # 保存模型信息
    def _save_info(self):
        state = {
            'epoch': self.epochs,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        return state

    # 保存模型以及词向量
    def save(self):
        id_word = {value:key for key ,value in self.vocab.items()}
        state = self._save_info()
        torch.save(state, self.model_save_path)
        self.logger.info('saving word2vec vector')
        metrix = self.model.v_embedding_matrix.weight.data
        with open(self.vector_save_path, "w", encoding="utf-8") as f:
            if self.device=='cpu':
                vector = metrix.numpy()
            else:
                vector = metrix.cpu().numpy()
            for i in tqdm(range(len(vector)),desc = 'save vector'):
                word  = id_word[i]
                s_vec = vector[i]
                s_vec = [str(s) for s in s_vec.tolist()]
                write_line = word + " " + " ".join(s_vec)+"\n"
                f.write(write_line)

    # epoch训练
    def _train_epoch(self):
        self.model.train()
        i = 0
        if self.device == 'cpu':
            input_type = torch.LongTensor
        else:
            input_type = torch.cuda.LongTensor
        train_examples = self.train_data.make_iter()
        for pos_u,pos_v,neg_u,neg_v in train_examples:
            start = time.time()
            pos_u = input_type(pos_u).to(self.device)
            pos_v = input_type(pos_v).to(self.device)
            neg_u = input_type(neg_u).to(self.device)
            neg_v = input_type(neg_v).to(self.device)
            self.optimizer.zero_grad()
            loss = self.model(pos_u, pos_v, neg_u, neg_v)
            loss.backward()
            self.optimizer.step()
            i += 1
            if self.verbose >= 1:
                self.progressbar.step(batch_idx=i,loss =loss.item(),
                                      use_time=time.time() - start)
    #训练
    def train(self):
        for epoch in range(self.start_epoch,self.start_epoch+self.epochs):
            print("----------------- training start -----------------------")
            print("Epoch {i}/{epochs}......".format(i=epoch, epochs=self.start_epoch+self.epochs -1))
            self._train_epoch()
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch)
            self.save()







