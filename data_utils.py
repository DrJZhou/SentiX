import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
# bert_path = "/gruntdata/zhoujie/bert_model"
bert_path = "D:/Projects/bert_model"


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name=""):
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class PreTrainDataset(Dataset):
    def __init__(self, data_set, tokenizer, train_or_test='train'):
        self.tokenizer = tokenizer
        path_save = "./data/sentiment_analysis/{}/{}.pkl".format(data_set, train_or_test)
        if os.path.exists(path_save):
            with open(path_save, 'rb') as fr:
                self.data = pickle.load(fr)
            return
        fname = "./data/sentiment_analysis/{}/{}.txt".format(data_set, train_or_test)
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        all_data = []
        for i in range(0, len(lines)):
            tmp = lines[i].strip().split("\t")
            text = tmp[0].strip()
            sentiment = int(tmp[1])
            text = " ".join(tokenizer.tokenizer.tokenize(text)[:self.tokenizer.max_seq_len-2])
            text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text + " [SEP]")

            data = {
                'text_bert_indices': text_bert_indices,
                'sentiment': sentiment,
            }
            all_data.append(data)
        with open(path_save, 'wb') as fr_to:
            pickle.dump(all_data, fr_to)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    tokenizer = Tokenizer4Bert(100, "")
    dataset = PreTrainDataset('sst-5', tokenizer, train_or_test='train')
