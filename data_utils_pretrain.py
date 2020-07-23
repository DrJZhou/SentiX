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
        self.random_sentiment = 0.1
        self.random_emoji = 0.5
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
                path_save = "./data/processed/{}_Rating.pkl".format(name)
                if os.path.exists(path_save):
                    with open(path_save, 'rb') as fr:
                        data_all += pickle.load(fr)
            self.data = data_all
            return
        path_save = "./data/processed/{}_Rating.pkl".format(data_set)
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
            text_tokens = []

            # # in word-level
            # word_sentiment_labels = np.array([1]*tokenizer.max_seq_len, dtype='int') # sentiment of the word, 0: negative, 1: neutral, 2:positive
            # word_sentiment_word_labels = np.array([0]*tokenizer.max_seq_len, dtype='int') # which sentiment word, 0: not sentiment word
            # word_emoji_labels = np.array([0]*tokenizer.max_seq_len, dtype='int') # which emoji, 0: not emoji
            #
            # # in sentence-level
            # sentence_sentiment_labels = np.array([0]*sentiment_tokenizer.sentiment_class, dtype='float64')
            # sentence_emoji_labels = np.array([0]*emoji_tokenizer.emoji_class, dtype='float64')
            # print(tmp)

            for index, token in enumerate(tmp):
                if index >= tokenizer.max_seq_len:
                    break
                try:
                    word = token[:-6]
                    sentiment = int(token[-4])
                    emoji = int(token[-1])
                except:
                    print(token)
                    continue

                # if (emoji != 0) and (word in emoji_tokenizer.emoji2index):
                    # word_emoji_labels[index] = emoji_tokenizer.emoji2index[word]
                    # sentence_emoji_labels[emoji_tokenizer.emoji2index[word]-1] = 1
                    # word = emoji_tokenizer.emoji2word[word]
                # elif (sentiment != 1) and (word in sentiment_tokenizer.sentiment2index):
                    # word_sentiment_labels[index] = sentiment
                    # # print(index, token, word, sentiment, emoji)
                    # word_sentiment_word_labels[index] = sentiment_tokenizer.sentiment2index[word]
                    # sentence_sentiment_labels[sentiment_tokenizer.sentiment2index[word]-1] = 1
                text_tokens.append("{}#{}#{}".format(word, sentiment, emoji))

            text = " ".join(text_tokens[:self.tokenizer.max_seq_len])
            # print(text)
            # print(word_sentiment_labels)
            # text = " ".join(tokenizer.tokenizer.tokenize(text)[:500])
            # text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text + " [SEP]")

            # data = {
            #     "text": text,
            #     # 'text_bert_indices': text_bert_indices,
            #     # 'word_sentiment_labels': word_sentiment_labels,
            #     # 'word_sentiment_word_labels': word_sentiment_word_labels,
            #     # 'word_emoji_labels': word_emoji_labels,
            #     # 'sentence_sentiment_labels': sentence_sentiment_labels,
            #     # 'sentence_emoji_labels': sentence_emoji_labels,
            # }
            all_data.append(text+"\t"+rating)
        with open(path_save, 'wb') as fr_to:
            pickle.dump(all_data, fr_to)
        self.data = all_data

    def __getitem__(self, index):
        text_rating = self.data[index].split("\t")
        # print(text_rating)
        rating = int(text_rating[-1]) - 1
        text = text_rating[0]
        text_tokens = text.split(" ")
        # print(text)
        tmp = {}
        sentence_sentiment_labels = np.array([0] * self.sentiment_tokenizer.sentiment_class, dtype='float64')
        sentence_emoji_labels = np.array([0] * self.emoji_tokenizer.emoji_class, dtype='float64')
        word_sentiment_labels = np.array([1] * self.tokenizer.max_seq_len,
                                         dtype='int')  # sentiment of the word, 0: negative, 1: neutral, 2:positive
        word_sentiment_word_labels = np.array([0] * self.tokenizer.max_seq_len,
                                              dtype='int')  # which sentiment word, 0: not sentiment word
        word_emoji_labels = np.array([0] * self.tokenizer.max_seq_len, dtype='int')  # which emoji, 0: not emoji
        num = 1
        text_token_tmp = ['[CLS]']
        for i in range(len(text_tokens)):
            token_sentiment_emoji = text_tokens[i]
            token = token_sentiment_emoji[:-4]
            sentiment = int(token_sentiment_emoji[-3])
            emoji = int(token_sentiment_emoji[-1])
            if num >= self.tokenizer.max_seq_len-2:
                break
            if emoji == 1:
                sentence_emoji_labels[self.emoji_tokenizer.emoji2index[token]-1] = 1
                emoji_random_number = np.random.random()
                if emoji_random_number < self.random_emoji:
                    text_token_tmp.append("[MASK]")
                else:
                    text_token_tmp.append(token)
                word_emoji_labels[num] = self.emoji_tokenizer.emoji2index[token]
                num += 1
            elif sentiment != 1:
                sentence_sentiment_labels[self.sentiment_tokenizer.sentiment2index[token]-1] = 1
                sentiment_random_number = np.random.random()
                if sentiment_random_number < self.random_sentiment:
                    text_token_tmp.append("[MASK]")
                    word_sentiment_labels[num] = sentiment
                    word_sentiment_word_labels[num] = self.sentiment_tokenizer.sentiment2index[token]
                    num += 1
                else:
                    word_piece = self.tokenizer.tokenizer.tokenize(token)
                    if num + len(word_piece) >= self.tokenizer.max_seq_len-2:
                        break
                    for word in word_piece:
                        text_token_tmp.append(word)
                        word_sentiment_labels[num] = sentiment
                        word_sentiment_word_labels[num] = self.sentiment_tokenizer.sentiment2index[token]
                        num += 1
            else:
                normal_random_number = np.random.random()
                if normal_random_number < 0.05:
                    text_token_tmp.append("[MASK]")
                    num += 1
                else:
                    word_piece = self.tokenizer.tokenizer.tokenize(token)
                    if num + len(word_piece) >= self.tokenizer.max_seq_len-2:
                        break
                    for word in word_piece:
                        text_token_tmp.append(word)
                        num += 1
        text_token_tmp.append("[SEP]")
        text_bert_indices = self.tokenizer.text_to_sequence(" ".join(text_token_tmp[: self.tokenizer.max_seq_len]))
        # tmp['text'] = " ".join(text_token_tmp[: self.tokenizer.max_seq_len])
        tmp['text_bert_indices'] = text_bert_indices
        tmp['word_emoji_labels'] = word_emoji_labels
        tmp['word_sentiment_labels'] = word_sentiment_labels
        tmp['word_sentiment_word_labels'] = word_sentiment_word_labels
        tmp['sentence_sentiment_labels'] = sentence_sentiment_labels
        tmp['sentence_emoji_labels'] = sentence_emoji_labels
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
#     for name in name_list[start_index: start_index+10]:
#         dataset = PreTrainDataset(name, tokenizer, emoji_tokenizer, sentiment_tokenizer)