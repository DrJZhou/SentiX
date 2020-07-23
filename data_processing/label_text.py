'''
remove the words that occur less than 10

'''
from utils import judge_sentiment, judge_emotion
import numpy as np

# statistic the word num of all the dataset
def statistic_word_num():
    name_list = ["All_Beauty", "AMAZON_FASHION", "Appliances", "Arts_Crafts_and_Sewing", "Automotive", "Books",
                 "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music",
                 "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food", "Home_and_Kitchen",
                 "Industrial_and_Scientific", "Kindle_Store", "Luxury_Beauty", "Magazine_Subscriptions",
                 "Movies_and_TV", "Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies",
                 "Prime_Pantry", "Software", "Sports_and_Outdoors", "Tools_and_Home_Improvement", "Toys_and_Games",
                 "Video_Games"]
    word_num = {}
    for name in name_list:
        input_file = "../data/amazon/{}.csv".format(name)
        fr = open(input_file, "r", encoding='utf-8', errors='ignore')
        try:
            for line in fr.readlines():
                tokens = line.strip().split(" ")
                for word in tokens:
                    word_num[word] = word_num.get(word, 0) + 1
        except:
            pass
        fr.close()

    input_file = "../data/yelp_dataset_clean.csv"
    fr = open(input_file, "r", encoding='utf-8', errors='ignore')
    try:
        for line in fr.readlines():
            tokens = line.strip().split(" ")
            for word in tokens:
                word_num[word] = word_num.get(word, 0) + 1
    except:
        pass
    fr.close()

    outfile = "../data/voc.csv"
    fr_to = open(outfile, 'w', encoding='utf-8')
    word_num = sorted(word_num.items(), key=lambda item: item[1], reverse=True)
    for e in word_num:
        fr_to.write("{},{}\n".format(e[0], e[1]))
    fr_to.close()


# load the top_k words
def load_voc():
    voc_set = {}
    input_file = "../data/voc.csv"
    fr = open(input_file, 'r', encoding='utf-8')
    for line in fr.readlines():
        data = line.strip().split(",")
        word = ",".join(data[:-1])
        num = int(data[-1])
        if num < 20:
            continue
        voc_set[word] = 1
    fr.close()
    return voc_set


# remove the low frequency word and label the sentiment word and emoji
def label_data():
    voc_set = load_voc()
    name_list = ["All_Beauty", "AMAZON_FASHION", "Appliances", "Arts_Crafts_and_Sewing", "Automotive", "Books",
                 "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music",
                 "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food", "Home_and_Kitchen",
                 "Industrial_and_Scientific", "Kindle_Store", "Luxury_Beauty", "Magazine_Subscriptions",
                 "Movies_and_TV", "Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies",
                 "Prime_Pantry", "Software", "Sports_and_Outdoors", "Tools_and_Home_Improvement", "Toys_and_Games",
                 "Video_Games"]
    now_index = 0
    for name in name_list[now_index:now_index+5]:
        input_file = "../data/amazon/{}_Rating.csv".format(name)
        fr = open(input_file, "r", encoding='utf-8', errors='ignore')
        output_file = "../data/labeled_data/{}_Rating_labeled.csv".format(name)
        fr_to = open(output_file, "w", encoding='utf-8', errors='ignore')
        try:
            for line in fr.readlines():
                data_tmp = line.strip().split("\t")
                text = data_tmp[0]
                rating = int(data_tmp[-1])
                tokens = text.strip().split(" ")
                tokens_after_remove = []
                for word in tokens:
                    if word in voc_set:
                        tokens_after_remove.append(word)
                sentiment_labels = judge_sentiment(tokens_after_remove)
                emoticon_labels = judge_emotion(tokens_after_remove)
                if np.sum(sentiment_labels == 1) == len(sentiment_labels) and np.sum(emoticon_labels) == 0:
                    continue
                tmp = ""
                for i in range(len(tokens_after_remove)):
                    tmp += "{}##{}##{}\t".format(tokens_after_remove[i], sentiment_labels[i], emoticon_labels[i])
                fr_to.write(tmp.strip()+"\t{}\n".format(rating))
        except:
            pass
        fr.close()
        fr_to.close()

    input_file = "../data/yelp_Rating.csv"
    fr = open(input_file, "r", encoding='utf-8', errors='ignore')
    output_file = "../data/labeled_data/yelp_Rating_labeled.csv"
    fr_to = open(output_file, "w", encoding='utf-8', errors='ignore')
    # try:
    for line in fr.readlines():
        data_tmp = line.strip().split("\t")
        text = data_tmp[0]
        rating = int(data_tmp[-1])
        tokens = text.strip().split(" ")
        tokens_after_remove = []
        for word in tokens:
            if word in voc_set:
                tokens_after_remove.append(word)
        # print(tokens_after_remove)
        sentiment_labels = judge_sentiment(tokens_after_remove)
        emoticon_labels = judge_emotion(tokens_after_remove)
        if np.sum(sentiment_labels == 1) == len(sentiment_labels) and np.sum(emoticon_labels) == 0:
            continue
        tmp = ""
        for i in range(len(tokens_after_remove)):
            tmp += "{}##{}##{}\t".format(tokens_after_remove[i], sentiment_labels[i], emoticon_labels[i])
        fr_to.write(tmp.strip() + "\t{}\n".format(rating))
        # print(tmp)
    # except:
    #     pass
    fr.close()
    fr_to.close()



if __name__ == '__main__':
    # statistic_word_num()
    label_data()