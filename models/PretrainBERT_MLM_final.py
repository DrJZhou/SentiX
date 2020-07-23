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
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def cal_loss(self, word, labels):
        loss1 = self.criterion(word, labels[0])
        loss = loss1
        return loss

    def forward(self, inputs, labels=None):
        text_bert_indices = inputs[0]
        last_hidden_states, pooled_output = self.bert(text_bert_indices)
        last_hidden_states = self.dropout(last_hidden_states)
        word = self.word_dense(last_hidden_states)
        word = word.view(-1, word.size(2))
        if labels is None:
            loss = None
        else:
            loss = self.cal_loss(word, labels)
        return word, loss
