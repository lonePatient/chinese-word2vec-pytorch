#encoding:utf-8
import os
import random
import torch
import json
import pickle
import numpy as np

# 设置seed环境
def seed_everything(seed = 1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# batch的数据处理
def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    transposed = zip(*batch)
    lbd = lambda batch:torch.cat([torch.from_numpy(b).long() for b in batch])
    return [lbd(samples) for samples in transposed]

class AverageMeter(object):
    '''
    computes and stores the average and current value
    '''
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val,n = 1):
        self.val  = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def json_write(data,filename):
    with open(filename,'w') as f:
        json.dump(data,f)

def json_read(filename):
    with open(filename,'r') as f:
        return json.load(f)

def text_write(file,data):
    with open(file, 'w') as fw:
        for sent in data:
            sent = ' '.join([str(s) for s in sent])
            fw.write(sent+"\n")

def pkl_read(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

def pkl_write(filename,data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


