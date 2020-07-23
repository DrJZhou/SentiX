import os
import spacy
import numpy as np
spacy_en = spacy.load("en")
import pandas as pd
# import sys
# from importlib import reload
# reload(sys)  # Reload does the trick!
# sys.setdefaultencoding('UTF8')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def processing_multidomain():

    pass

def processing_SST1_2(dataset_name="SST-1"):
    SST_path = "D:/datasets/sentiment_analysis/SST-2/"
    path_to = "../data/sentiment_analysis/{}/".format(dataset_name)
    for fname_input, fname_output in zip(['train', 'test', 'dev'], ['train', 'test', 'val']):
        input_file = open(SST_path + "sst_{}.txt".format(fname_input), 'r', encoding='utf-8')
        output_file = open(path_to + "{}.txt".format(fname_output), 'w', encoding='utf-8')
        for line in input_file.readlines():
            data = line.strip().split("\t")
            label = int(data[0][-1])-1
            text = " ".join(tokenize_en(data[1]))
            if dataset_name == "SST-1":
                output_file.write("{}\t{}\n".format(text, label))
            else:
                if label != 2:
                    output_file.write("{}\t{}\n".format(text, int(label/3)))
        output_file.close()
        input_file.close()


def processing_SST(dataset_name='SST-1'):
    SST_path = "D:/datasets/sentiment_analysis/stanfordSentimentTreebank/"
    datasetSentences = pd.read_csv(SST_path + 'datasetSentences.txt', sep='\t', encoding='utf-8')
    dictionary = pd.read_csv(SST_path + 'dictionary.txt', sep='|', header=None, names=['sentence', 'phrase ids'], encoding='utf-8')
    datasetSplit = pd.read_csv(SST_path + 'datasetSplit.txt', sep=',', encoding='utf-8')
    sentiment_labels = pd.read_csv(SST_path + 'sentiment_labels.txt', sep='|', encoding='utf-8')

    # 将多个表进行内连接合并
    dataset = pd.merge(pd.merge(pd.merge(datasetSentences, datasetSplit), dictionary), sentiment_labels)
    print(datasetSplit[datasetSplit['splitset_label'] == 1])
    def labeling(data_name, sentiment_value):
        '''
        将情感值转为标签

        :param data_name: SST-1/SST-2
        :param sentiment_value: sentiment_value
        :return: label
        '''
        if data_name == 'SST-1':
            if sentiment_value <= 0.2:
                return 0  # very negative
            elif sentiment_value <= 0.4:
                return 1  # negative
            elif sentiment_value <= 0.6:
                return 2  # neutral
            elif sentiment_value <= 0.8:
                return 3  # positive
            elif sentiment_value <= 1:
                return 4  # very positive
        else:
            if sentiment_value <= 0.4:
                return 0  # negative
            elif sentiment_value > 0.6:
                return 1  # positive
            else:
                return -1  # drop neutral

    # 将情感值转为标签
    dataset['sentiment_label'] = dataset['sentiment values'].apply(lambda x: labeling(dataset_name, x))
    dataset = dataset[dataset['sentiment_label'] != -1]

    path_to = "../data/sentiment_analysis/{}/".format(dataset_name)
    if not os.path.exists(path_to):
        os.mkdir(path_to)

    output_file_1 = path_to + "train.txt"
    output_file_2 = path_to + "test.txt"
    output_file_3 = path_to + "val.txt"

    def filter_punctuation(s):
        s = " ".join(tokenize_en(s)).strip().lower()
        return s

    # 对句子进行预处理
    dataset['sentence'] = dataset['sentence'].apply(lambda s: filter_punctuation(s))
    # 保存处理好的数据集
    # train
    dataset[dataset['splitset_label'] == 1][['sentence', 'sentiment_label']].to_csv(
        output_file_1, index=False, sep='\t', header=False)
    # test
    dataset[dataset['splitset_label'] == 2][['sentence', 'sentiment_label']].to_csv(
        output_file_2, index=False, sep='\t', header=False)
    # dev
    dataset[dataset['splitset_label'] == 3][['sentence', 'sentiment_label']].to_csv(
        output_file_3, index=False, sep='\t', header=False)


def processing_IMDB():
    path = "D:/datasets/sentiment_analysis/aclImdb/"
    output_path = '../data/sentiment_analysis/IMDB/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path_train = open(output_path+"train.txt", 'w', encoding='utf-8')
    output_path_test = open(output_path + "test.txt", 'w', encoding='utf-8')
    for name1 in ['train', 'test']:
        for name2 in ['pos', 'neg']:
            path_ = path + "{}/{}/".format(name1, name2)
            fnames = os.listdir(path_)
            for fname in fnames:
                fr = open(path_+fname, 'r', encoding='utf-8')
                sentence = fr.read().strip().replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("<br />", " ")
                sentence = " ".join(tokenize_en(sentence)).lower().strip()
                if name1 == 'train':
                    if name2 == 'pos':
                        output_path_train.write("{}\t{}\n".format(sentence, 1))
                    else:
                        output_path_train.write("{}\t{}\n".format(sentence, 0))
                else:
                    if name2 == 'pos':
                        output_path_test.write("{}\t{}\n".format(sentence, 1))
                    else:
                        output_path_test.write("{}\t{}\n".format(sentence, 0))
    output_path_train.close()
    output_path_test.close()


def processing_Yelp():
    Yelp_path = "D:/datasets/sentiment_analysis/yelp_review_polarity_csv/"
    path_to = "../data/sentiment_analysis/Yelp/"

    input_file_train = Yelp_path + "train.csv"
    output_file_train = path_to + "train.txt"
    fr = open(input_file_train, 'r', encoding='utf-8')
    f_to = open(output_file_train, 'w', encoding='utf-8')
    for line in fr.readlines():
        data = line.strip().split(",")
        polarity = int(data[0].replace('"', "")) - 1
        sentence = ",".join(data[1:])[1: -1]
        sentence = " ".join(sentence.replace("\\n", " ").replace('\\"', " ").split(" "))
        sentence = " ".join(tokenize_en(sentence)).lower().strip()
        f_to.write("{}\t{}\n".format(sentence, polarity))
    fr.close()
    f_to.close()

    input_file_test = Yelp_path + "test.csv"
    output_file_test = path_to + "test.txt"
    fr = open(input_file_test, 'r', encoding='utf-8')
    f_to = open(output_file_test, 'w', encoding='utf-8')
    for line in fr.readlines():
        data = line.strip().split(",")
        polarity = int(data[0].replace('"', "")) - 1
        sentence = ",".join(data[1:])[1: -1]
        sentence = " ".join(sentence.replace("\\n", " ").replace('\\"', " ").split(" "))
        sentence = " ".join(tokenize_en(sentence)).lower().strip()
        f_to.write("{}\t{}\n".format(sentence, polarity))
    fr.close()
    f_to.close()


def processing_multidomain():
    def read_data(fname, sentiment=0):
        data_all = []
        fr = open(fname, 'r', encoding='utf-8')
        text = ""
        flag = 0
        for line in fr.readlines():
            if line.strip() == "<review_text>":
                text = ""
                flag = 1
            elif line.strip() == "</review_text>":
                flag = 0
                text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
                text = " ".join(tokenize_en(text)).lower()
                text = " ".join(text.split(" "))
                data_all.append([text, sentiment])
            elif flag == 1:
                text += line.strip()
        return data_all

    input_path = "D:/datasets/sentiment_analysis/sorted_data_acl/"
    for domain in ['books', 'dvd', 'electronics', 'kitchen_&_housewares']:
        input_fnmae_1 = "{}{}/positive.review".format(input_path, domain)
        input_fnmae_2 = "{}{}/negative.review".format(input_path, domain)
        data_all = read_data(input_fnmae_1, sentiment=1)
        data_all += read_data(input_fnmae_2, sentiment=0)
        np.random.shuffle(data_all)

        output_path = "../data/multidomain/{}/".format(domain)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        f_to = open(output_path+"{}.txt".format(domain), 'w', encoding='utf-8')
        for raw in data_all:
            f_to.write("{}\t{}\n".format(raw[0], raw[1]))
        f_to.close()


if __name__ == '__main__':
    # processing_SST1_2(dataset_name='SST-1')
    # processing_SST1_2(dataset_name='SST-2')
    # processing_IMDB()
    # processing_Yelp()
    processing_multidomain()