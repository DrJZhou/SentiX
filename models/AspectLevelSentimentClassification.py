import torch
import torch.nn as nn
import torch.nn.functional as F


class AspectLevelSentimentClassification(nn.Module):
    def __init__(self, bert, opt):
        super(AspectLevelSentimentClassification, self).__init__()
        self.bert = bert
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.linear = nn.Linear(opt.bert_dim*2, opt.sentiment_class)

    def forward(self, inputs):
        text_bert_indices, position_indices = inputs[0], inputs[1]
        last_hidden_state, pooled_output = self.bert(text_bert_indices)
        aspect_representation = None
        for i in range(pooled_output.size(0)):
            aspect_ = torch.mean(last_hidden_state[i, position_indices[i, 0]:position_indices[i, 1], :], dim=0)
            if aspect_representation is None:
                aspect_representation = aspect_
            else:
                aspect_representation = torch.cat((aspect_representation, aspect_))
        aspect_representation = aspect_representation.view(pooled_output.size(0), pooled_output.size(1))
        pooled_output = torch.cat([pooled_output, aspect_representation], dim=-1)
        pooled_output = self.dropout(pooled_output)
        output = self.linear(pooled_output)
        return output
