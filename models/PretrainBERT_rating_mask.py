import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainBERT(nn.Module):
    def __init__(self, bert, opt):
        super(PretrainBERT, self).__init__()
        self.bert = bert
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.word_dense = nn.Linear(opt.bert_dim, 30522)
        self.rating_dense = nn.Linear(opt.bert_dim, 5)

    def forward(self, inputs):
        text_bert_indices = inputs[0]
        last_hidden_states, pooled_output = self.bert(text_bert_indices)
        pooled_output = self.dropout(pooled_output)
        last_hidden_states = self.dropout(last_hidden_states)
        word = self.word_dense(last_hidden_states)
        rating = self.rating_dense(pooled_output)
        return word, rating
