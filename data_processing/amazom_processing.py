'''
deal with the amazon dataset
'''

import json
import gzip
from utils import text_processing, judge_emoji
# base_path = "D:/datasets/amazon/"
base_path = "/gruntdata/zhoujie/dataset/amazon/"


# parse the json zip dataset
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        # print(l)
        yield json.loads(l)


#clean the dataset (for a domain)
def processing_one_category(input_path, output_path):
    f = open(output_path, 'w', encoding="utf-8", errors='ignore')
    line_num = 0
    for l in parse(input_path):
        line_num += 1
        if "reviewText" in l and "overall" in l:
            text = l["reviewText"].strip().replace("\t", " ")
            rating = int(l['overall'])
            text_token = text_processing(text)
            text = " ".join(text_token).strip()
            if len(text_token) > 5:
                try:
                    f.write("{}\t{}\n".format(text, rating))
                except:
                    pass
        # if line_num % 10000 == 0:
        #     print(line_num)
    f.close()


# stastistic the emoji information of each domain
def statistic_emoticon(input_file_list):
    for name in input_file_list:
        filename_input = "../data/amazon/{}.csv".format(name)
        fr = open(filename_input, "r", encoding='utf-8')
        emoticon_num = {}
        for line in fr.readlines():
            text_token = line.strip().split(" ")
            for w in text_token:
                if judge_emoji(w):
                    emoticon_num[w] = emoticon_num.get(w, 0) + 1
        fr.close()
        outfile = "../data/amazon/emoticon_{}.csv".format(name)
        fr_to = open(outfile, 'w', encoding='utf-8')
        emoticon_num_sort = sorted(emoticon_num.items(), key=lambda item: item[1], reverse=True)
        print(emoticon_num_sort[: 100])
        print(emoticon_num)
        for e in emoticon_num_sort:
            fr_to.write("{},{}\n".format(e[0], e[1]))
        fr_to.close()


def main():
    name_list = ["All_Beauty", "AMAZON_FASHION", "Appliances", "Arts_Crafts_and_Sewing", "Automotive", "Books",
                 "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music",
                 "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food", "Home_and_Kitchen",
                 "Industrial_and_Scientific", "Kindle_Store", "Luxury_Beauty", "Magazine_Subscriptions",
                 "Movies_and_TV", "Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies",
                 "Prime_Pantry", "Software", "Sports_and_Outdoors", "Tools_and_Home_Improvement", "Toys_and_Games",
                 "Video_Games"]
    start_index = 0
    for name in name_list[start_index: start_index+5]:
        processing_one_category("{}{}.json.gz".format(base_path, name), "../data/amazon/{}_Rating.csv".format(name))

    # for name in name_list[10: 20]:
    #     processing_one_category("{}{}.json.gz".format(base_path, name), "../data/amazon/{}.csv".format(name))

    # statistic_emoticon(name_list[20:])


if __name__ == '__main__':
    main()