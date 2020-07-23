import pickle
import os
import numpy as np
name_list = ["All_Beauty", "AMAZON_FASHION", "Appliances", "Arts_Crafts_and_Sewing", "Automotive", "Books",
                 "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music",
                 "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food", "Home_and_Kitchen",
                 "Industrial_and_Scientific", "Kindle_Store", "Luxury_Beauty", "Magazine_Subscriptions",
                 "Movies_and_TV", "Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies",
                 "Prime_Pantry", "Software", "Sports_and_Outdoors", "Tools_and_Home_Improvement", "Toys_and_Games",
                 "Video_Games", 'yelp']

each_class_num = 500000
min_len = 50
max_len = 510
for name in name_list:
    path_read = "../data/processed/{}.pkl".format(name)
    path_save = "../data/processed/{}_balance_{}.pkl".format(name, each_class_num)
    if os.path.exists(path_read) and not os.path.exists(path_save):
        with open(path_read, 'rb') as fr:
            data = pickle.load(fr)
            np.random.shuffle(data)
            balance_data = [[]]*5
            data_num = [0]*5
            for index in range(len(data)):
                content = data[index]
                text_rating = content.split("\t")
                text_tokens = text_rating[0].split(" ")
                if len(text_tokens) > max_len or len(text_tokens) < min_len:
                    continue
                rating = int(text_rating[-1]) - 1
                if data_num[rating] < each_class_num:
                    data_num[rating] += 1
                    balance_data[rating].append(content)
                if np.sum(data_num) == each_class_num*5:
                    break
            min_rating_num = np.min(data_num)
            all_data = []
            final_num = min(min_rating_num*2, each_class_num)
            for i in range(5):
                all_data += balance_data[i][: final_num]
            np.random.shuffle(all_data)
            print(len(all_data))
            with open(path_save, 'wb') as f_to:
                pickle.dump(all_data, f_to)
