name_list = ["All_Beauty", "AMAZON_FASHION", "Appliances", "Arts_Crafts_and_Sewing", "Automotive", "Books",
                 "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music",
                 "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food", "Home_and_Kitchen",
                 "Industrial_and_Scientific", "Kindle_Store", "Luxury_Beauty", "Magazine_Subscriptions",
                 "Movies_and_TV", "Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies",
                 "Prime_Pantry", "Software", "Sports_and_Outdoors", "Tools_and_Home_Improvement", "Toys_and_Games",
                 "Video_Games", 'yelp']
for data_set in name_list:
    fname = "../data/labeled_data/{}_Rating_labeled.csv".format(data_set)
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    print(data_set, len(lines))
    del lines
