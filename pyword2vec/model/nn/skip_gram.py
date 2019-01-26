#encoding:Utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGram(torch.nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(SkipGram, self).__init__()
        initrange = 0.5 / embedding_dim
        self.u_embedding_matrix = nn.Embedding(vocab_size,embedding_dim)
        self.u_embedding_matrix.weight.data.uniform_(-initrange,initrange)
        self.v_embedding_matrix = nn.Embedding(vocab_size,embedding_dim)
        self.v_embedding_matrix.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v,neg_u, neg_v):
        embed_pos_u = self.v_embedding_matrix(pos_u)
        embed_pos_v = self.u_embedding_matrix(pos_v)
        score = torch.mul(embed_pos_u, embed_pos_v)
        score = torch.sum(score,dim = 1)
        log_target = F.logsigmoid(score).squeeze()

        embed_neg_u = self.u_embedding_matrix(neg_u)
        embed_neg_v = self.v_embedding_matrix(neg_v)

        neg_score = torch.mul(embed_neg_u,embed_neg_v)
        neg_score = torch.sum(neg_score, dim=1)
        sum_log_sampled = F.logsigmoid(-1 * neg_score).squeeze()

        loss = log_target.sum() + sum_log_sampled.sum()
        loss = -1 * loss
        return loss
