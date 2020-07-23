# -*- coding: utf-8 -*-
"""
Convert the Yelp Dataset Challenge dataset from json format to csv.
"""
import argparse
import collections
import csv
import re
import simplejson as json
from utils import text_processing, judge_emoji, judge_url


# read the reviews of yelp dataset
def get_superset_of_column_names_from_file(json_file_path, csv_file_path):
    column_names = set()
    f_to = open(csv_file_path, "w")
    with open(json_file_path) as fin:
        for line in fin:
            line_contents = json.loads(line)
            # print(line_contents)
            text = line_contents["text"].strip().replace("\t", " ")
            text_token = text_processing(text)
            text = " ".join(text_token).strip()
            rating = int(line_contents['stars'])
            if len(text_token) > 5:
                f_to.write("{}\t{}\n".format(text, rating))
    f_to.close()


# clean the review data
def clean(csv_file_path, csv_file_clean_path):
    fr = open(csv_file_path, 'r', encoding="utf-8", errors="ignore")
    dataset = set()
    line = fr.readline()
    while line:
        text = line.strip()
        text_token = text_processing(text)
        text = " ".join(text_token).strip()
        if len(text.strip()) > 0:
            dataset.add(text)
        line = fr.readline()

    fr = open(csv_file_clean_path, 'w', encoding='utf-8')
    for text in dataset:
        fr.write(text+"\n")
    fr.close()


# statistic the emoji information of yelp dataset
def processing(csv_file_clean_path, csv_file_emoticon_path):
    fr = open(csv_file_clean_path, 'r', encoding="utf-8", errors="ignore")
    line = fr.readline()
    # emoji, unicode_emoji = load_emoji()
    # emoticon = load_emoticon()
    fr_to = open(csv_file_emoticon_path, 'w', encoding='utf-8')
    emoticon_num = {}
    line_num = 0
    while line:
        text = line.strip()
        text_token = text.split(" ")
        for w in text_token:
            if judge_emoji(word=w):
                emoticon_num[w] = emoticon_num.get(w, 0) + 1
        line_num += 1
        # if line_num > 10000:
        #     break
        line = fr.readline()
    emoticon_num_sort = sorted(emoticon_num.items(), key=lambda item: item[1], reverse=True)
    print(emoticon_num_sort[: 100])
    print(emoticon_num)
    for e in emoticon_num_sort:
        fr_to.write("{},{}\n".format(e[0], e[1]))
    fr_to.close()


if __name__ == '__main__':
    # json_file = "D:/datasets/yelp_dataset/yelp_dataset"
    json_file = "/gruntdata/zhoujie/dataset/yelp_academic_dataset_review.json"
    csv_file = '../data/yelp_Rating.csv'
    get_superset_of_column_names_from_file(json_file, csv_file)
    # clean("../data/yelp_dataset.csv", "../data/yelp_dataset_clean.csv")
    # processing("../data/yelp_dataset_clean.csv", "../data/yelp_dataset_emoticon.csv")
