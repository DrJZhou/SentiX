import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
bert_path = "/gruntdata/zhoujie/bert_model"


class EmojiTokenizer():
    def __init__(self, top_k=25):
        self.emoji2word = {}
        self.word2emoji = {}
        self.emoji2index = {}
        self.index2emoji = {}
        self.transfer_emoji(top_k=top_k)
        self.emoji_class = len(self.emoji2word)

    def transfer_emoji(self, top_k):
        fr = open("./data/emoticon_all.csv", 'r', encoding='utf-8')
        for index, line in enumerate(fr.readlines()[: top_k]):
            data = line.strip().split(",")
            emoji = ",".join(data[:-1])
            word = "[unused{}]".format(index+1)
            self.emoji2word[emoji] = word
            self.word2emoji[word] = emoji
            self.index2emoji[index+1] = emoji
            self.emoji2index[emoji] = index+1


class SentimentTokenizer():
    def __init__(self, top_k=None):
        self.sentiment2index = {}
        self.index2sentiment = {}
        self.sentiment_transfor(top_k=top_k)
        self.sentiment_class = len(self.sentiment2index)

    def sentiment_transfor(self, top_k=None):
        fr = open("data/sentiment_lexicon_all.csv", 'r', encoding='utf-8')
        lines = fr.readlines()
        fr.close()
        if top_k is None:
            top_k = len(lines)
        for index, line in enumerate(lines[:top_k]):
            data = line.strip().split(",")
            word = ",".join(data[:-1])
            self.sentiment2index[word] = index+1
            self.index2sentiment[index+1] = word


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
    def __init__(self, data_set, tokenizer, emoji_tokenizer, sentiment_tokenizer):
        self.tokenizer = tokenizer
        self.emoji_tokenizer = emoji_tokenizer
        self.sentiment_tokenizer = sentiment_tokenizer
        self.random_sentiment = 0.0
        self.random_emoji = 0.0
        CLS_SEP_id = self.tokenizer.tokenizer.convert_tokens_to_ids(['CLS', "SEP"])
        # random_word = 0.2
        if data_set == "data_all":
            name_list = ["All_Beauty", "AMAZON_FASHION", "Appliances", "Arts_Crafts_and_Sewing", "Automotive", "Books",
                         "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music",
                         "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food", "Home_and_Kitchen",
                         "Industrial_and_Scientific", "Kindle_Store", "Luxury_Beauty", "Magazine_Subscriptions",
                         "Movies_and_TV", "Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden",
                         "Pet_Supplies",
                         "Prime_Pantry", "Software", "Sports_and_Outdoors", "Tools_and_Home_Improvement",
                         "Toys_and_Games",
                         "Video_Games", 'yelp']
            data_all = []
            for name in name_list:
                path_save = "./data/processed/{}.pkl".format(name)
                if os.path.exists(path_save):
                    with open(path_save, 'rb') as fr:
                        data_all += pickle.load(fr)
            self.data = data_all
            return
        path_save = "./data/processed/{}.pkl".format(data_set)
        if os.path.exists(path_save):
            with open(path_save, 'rb') as fr:
                self.data = pickle.load(fr)
            return
        fname = "./data/labeled_data/{}_Rating_labeled.csv".format(data_set)
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        all_data = []
        for i in range(0, len(lines)):
            text_rating = lines[i].strip().split("\t")
            rating = text_rating[-1]
            tmp = text_rating[:-1]
            text_tokens = ["{}#{}#{}#{}".format(CLS_SEP_id[0], 1, -1, -1)]
            for index, token in enumerate(tmp):
                try:
                    word = token[:-6]
                    sentiment = int(token[-4])
                    emoji = int(token[-1])
                except:
                    continue
                emoji_id = -1
                sentiment_id = -1
                if emoji != 0:
                    emoji_id = emoji_tokenizer.emoji2index[word]-1
                    word = emoji_tokenizer.emoji2word[word]
                elif sentiment != 1:
                    sentiment_id = sentiment_tokenizer.sentiment2index[word] - 1

                word_piece = self.tokenizer.tokenizer.tokenize(word)
                word_piece_ids = self.tokenizer.tokenizer.convert_tokens_to_ids(word_piece)
                for index, piece in enumerate(word_piece_ids):
                    if index == 0:
                        text_tokens.append("{}#{}#{}#{}".format(piece, sentiment, sentiment_id, emoji_id))
                    else:
                        text_tokens.append("{}#{}#{}#{}".format(piece, sentiment, -1, -1))
                if len(text_tokens) >= 512:
                    break
            text_tokens = text_tokens[:self.tokenizer.max_seq_len-1] + ["{}#{}#{}#{}".format(CLS_SEP_id[1], 1, -1, -1)]
            all_data.append("{}\t{}".format(" ".join(text_tokens), rating))
        with open(path_save, 'wb') as fr_to:
            pickle.dump(all_data, fr_to)
        self.data = all_data

    def __getitem__(self, index):
        text_rating = self.data[index].split("\t")
        # print(text_rating)
        rating = int(text_rating[-1]) - 1
        text_tokens = text_rating[0].split(" ")
        # print(text)
        tmp = {}
        text_token_unmasked = []
        for i in range(len(text_tokens)):
            tt = text_tokens[i].split("#")
            token = int(tt[0])
            text_token_unmasked.append(token)

        text_bert_indices = pad_and_truncate(text_token_unmasked, self.tokenizer.max_seq_len)
        tmp['text_bert_indices'] = text_bert_indices
        tmp['rating'] = rating
        return tmp

    def __len__(self):
        return len(self.data)


# if __name__ == '__main__':
#     emoji_tokenizer = EmojiTokenizer()
#     sentiment_tokenizer = SentimentTokenizer()
#     tokenizer = Tokenizer4Bert(512, "")
#     name_list = ["All_Beauty", "AMAZON_FASHION", "Appliances", "Arts_Crafts_and_Sewing", "Automotive", "Books",
#                  "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music",
#                  "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food", "Home_and_Kitchen",
#                  "Industrial_and_Scientific", "Kindle_Store", "Luxury_Beauty", "Magazine_Subscriptions",
#                  "Movies_and_TV", "Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies",
#                  "Prime_Pantry", "Software", "Sports_and_Outdoors", "Tools_and_Home_Improvement", "Toys_and_Games",
#                  "Video_Games", 'yelp']
#     start_index = 0
#     for name in name_list[start_index: start_index+5]:
#         dataset = PreTrainDataset(name, tokenizer, emoji_tokenizer, sentiment_tokenizer)