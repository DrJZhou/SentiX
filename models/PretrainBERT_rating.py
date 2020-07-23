import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainBERT(nn.Module):
    def __init__(self, bert, opt):
        super(PretrainBERT, self).__init__()
        self.bert = bert
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.rating_dense = nn.Linear(opt.bert_dim, 5)

    def forward(self, inputs):
        text_bert_indices = inputs[0]
        _, pooled_output = self.bert(text_bert_indices)
        pooled_output = self.dropout(pooled_output)
        rating = self.rating_dense(pooled_output)
        return rating
