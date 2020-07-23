'''
loading the  emoji dataset set
Note: it is no use now.
'''


from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer
import re
tknzr = TweetTokenizer()


def load_emoji(path="../data/Emoji_Sentiment_Data_v1.0.csv"):
    fr = open(path, 'r', encoding="utf-8")
    flag = 0
    emoji = set()
    unicode_emoji = set()
    for line in fr.readlines():
        if flag == 0:
            flag = 1
            continue
        data = line.strip().split(",")
        emoji.add(data[0])
        unicode_emoji.add(data[1])
        flag += 1
        if flag > 30:
            break
        # print(data)
    return emoji, unicode_emoji


def load_emoticon():
    emoticon_list = [":)", ":(", ":((", ':">', ";)", ":D", ":->", ":P", "<3", "</3", ":0", "XD", ">:(", ">:D", "D:<", "=K", ":s", ";P", "=)", "-0)", ":-)",  ":^)", ":0", "D:", "(*_*)", "(T_T)", "T_T", "^^", "(x_x)", "(-_-:)", "(^^)", "(^.^)", ">.<", "o_0", "0.0", "e_e", "e.e", "*,..,*", "-O-", "-3-", "-w-", "'_'", ";_;", ":>", ".V.", "^_^'", "!>_<!", "<@>_____<@>;;", ";O;", "*u*", ":*", ":'(", ":/", "O:)", ":P", ":O", "&)", "^_^", ">:O", ":3", ">:(", "8|", "O.o", "-_-", "3:)", "<3", ":V", ":|]", "(^^^)", '<(")', "=D", ":>", "=D", ":]", "(*)", "B|", "8|", ";)", ";-)", ":-h", ":-w", "(:|", ">:O"]
    emoticon_set = {}
    # emoticon_token_set = {}
    for e in emoticon_list:
        emoticon_set[e] = 1
    return emoticon_set


# statistic top emoji
def load_top_emoji(top_k=100):
    emoji_num = {}
    name_list = ["All_Beauty", "AMAZON_FASHION", "Appliances", "Arts_Crafts_and_Sewing", "Automotive", "Books",
                 "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music",
                 "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food", "Home_and_Kitchen",
                 "Industrial_and_Scientific", "Kindle_Store", "Luxury_Beauty", "Magazine_Subscriptions",
                 "Movies_and_TV", "Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies",
                 "Prime_Pantry", "Software", "Sports_and_Outdoors", "Tools_and_Home_Improvement", "Toys_and_Games",
                 "Video_Games"]
    for name in name_list:
        input_file = "../data/amazon/emoticon_{}.csv".format(name)
        fr = open(input_file, 'r', encoding='utf-8')
        for line in fr.readlines():
            data = line.strip().split(",")
            emoji = ",".join(data[:-1])
            emoji_num[emoji] = emoji_num.get(emoji, 0) + int(data[-1])
        fr.close()
    fr = open("../data/yelp_dataset_emoticon.csv", 'r', encoding='utf-8')
    for line in fr.readlines():
        data = line.strip().split(",")
        emoji = ",".join(data[:-1])
        emoji_num[emoji] = emoji_num.get(emoji, 0) + int(data[-1])
    fr.close()
    outfile = "../data/emoticon_all.csv"
    fr_to = open(outfile, 'w', encoding='utf-8')
    emoticon_num_sort = sorted(emoji_num.items(), key=lambda item: item[1], reverse=True)
    print(emoticon_num_sort[: top_k])
    # print(emoji_num)
    for e in emoticon_num_sort[: top_k]:
        fr_to.write("{},{}\n".format(e[0], e[1]))
    fr_to.close()


if __name__ == '__main__':
    # load_emoji(path="../data/Emoji_Sentiment_Data_v1.0.csv")
    # emoticon = load_emoticon()
    load_top_emoji(top_k=1000)

