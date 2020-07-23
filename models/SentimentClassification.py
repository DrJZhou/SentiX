import torch
import torch.nn as nn
import torch.nn.functional as F


class SentimentClassification(nn.Module):
    def __init__(self, bert, opt):
        super(SentimentClassification, self).__init__()
        self.bert = bert
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.linear = nn.Linear(opt.bert_dim, opt.sentiment_class)

    def forward(self, inputs):
        text_bert_indices = inputs[0]
        last_hidden_states, pooled_output = self.bert(text_bert_indices)
        pooled_output = self.dropout(pooled_output)
        output = self.linear(pooled_output)
        return output
