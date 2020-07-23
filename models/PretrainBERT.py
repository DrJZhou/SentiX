import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainBERT(nn.Module):
    def __init__(self, bert, opt):
        super(PretrainBERT, self).__init__()
        self.bert = bert
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.word_sentiment_dense = nn.Linear(opt.bert_dim, 3)
        self.word_sentiment_word_dense = nn.Linear(opt.bert_dim, opt.sentiment_class+1)
        self.word_emoji_dense = nn.Linear(opt.bert_dim, opt.emoji_class+1)
        self.sentence_sentiment_dense = nn.Linear(opt.bert_dim, opt.sentiment_class)
        self.sentence_emoji_dense = nn.Linear(opt.bert_dim, opt.emoji_class)
        self.rating_dense = nn.Linear(opt.bert_dim, 5)

    def forward(self, inputs):
        text_bert_indices = inputs[0]
        last_hidden_states, pooled_output = self.bert(text_bert_indices)
        pooled_output = self.dropout(pooled_output)
        # last_hidden_states = last_hidden_states[text_bert_indices==103]
        if last_hidden_states.size(0) == 0:
            word_sentiment, word_sentiment_word, word_emoji = None, None, None
        else:
            last_hidden_states = self.dropout(last_hidden_states)
            word_sentiment = self.word_sentiment_dense(last_hidden_states)
            word_sentiment_word = self.word_sentiment_word_dense(last_hidden_states)
            word_emoji = self.word_emoji_dense(last_hidden_states)
        sentence_sentiment = self.sentence_sentiment_dense(pooled_output)
        sentence_emoji = self.sentence_emoji_dense(pooled_output)
        rating = self.rating_dense(pooled_output)
        if self.opt.circle_loss:
            pass
        else:
            sentence_sentiment = F.sigmoid(sentence_sentiment)
            sentence_emoji = F.sigmoid(sentence_emoji)
        return word_sentiment, word_sentiment_word, word_emoji, sentence_sentiment, sentence_emoji, rating
