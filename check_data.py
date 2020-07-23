import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy
import numpy as np

from transformers import BertModel

from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from data_utils_pretrain import Tokenizer4Bert, SentimentTokenizer, EmojiTokenizer, PreTrainDataset

emoji_tokenizer = EmojiTokenizer()
sentiment_tokenizer = SentimentTokenizer()
tokenizer = Tokenizer4Bert(512, "")
dataset = PreTrainDataset("All_Beauty", tokenizer, emoji_tokenizer, sentiment_tokenizer)
train_num = int(len(dataset.data)*0.9)
test_num = len(dataset.data) - train_num
trainset, testset = random_split(dataset, (train_num, test_num))

train_sampler = SequentialSampler(trainset)
train_data_loader = DataLoader(dataset=trainset, sampler=train_sampler, batch_size=8, num_workers=8)
for i_batch, sample_batched in enumerate(train_data_loader):
    if i_batch > 0:
        break
    text = sample_batched['text']
    text_bert_indices = sample_batched['text_bert_indices']
    word_emoji_labels = sample_batched['word_emoji_labels']
    word_sentiment_labels = sample_batched['word_sentiment_labels']
    word_sentiment_word_labels = sample_batched['word_sentiment_word_labels']
    sentence_sentiment_labels = sample_batched['sentence_sentiment_labels']
    sentence_emoji_labels = sample_batched['sentence_emoji_labels']
    for i in range(len(text_bert_indices)):
        if i > 0:
            continue
        print(text)
        print(text_bert_indices[i])
        print(word_sentiment_word_labels[i])
        print(word_sentiment_labels[i])
        print(word_emoji_labels[i])
        print(sentence_emoji_labels[i])
        print(sentence_sentiment_labels[i])