import torch
import torch.nn as nn
import torch.nn.functional as F


class AspectLevelSentimentClassification(nn.Module):
    def __init__(self, bert, opt):
        super(AspectLevelSentimentClassification, self).__init__()
        self.bert = bert
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.linear = nn.Linear(opt.bert_dim, opt.sentiment_class)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        _, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)
        pooled_output = self.dropout(pooled_output)
        output = self.linear(pooled_output)
        return output
