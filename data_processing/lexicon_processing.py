'''
deal with the original lexicon dataset
'''


def processing_lexicon_NRC(input_path, output_path):
    fr = open(input_path, "r", encoding='utf-8')
    negative_words = []
    positive_words = []
    for line in fr.readlines():
        data = line.strip().split("\t")
        if data[1] == "negative":
            if data[2] == "1":
                negative_words.append(data[0])
        if data[1] == "positive":
            if data[2] == "1":
                positive_words.append(data[0])
    fr_to = open(output_path, "w", encoding="utf-8")
    for word in positive_words:
        fr_to.write("{},1\n".format(word))
    for word in negative_words:
        fr_to.write("{},-1\n".format(word))
    print(len(positive_words))
    print(len(negative_words))


# load the lexicon
def load_lexicon():
    fr = open("../data/lexicon.csv", 'r', encoding='utf-8')
    lexicon_set = {}
    positive_words = set()
    negative_words = set()
    for line in fr.readlines():
        data = line.strip().split(",")
        word = data[0]
        sentiment = int(data[1])
        lexicon_set[word] = sentiment
        if sentiment == 1:
            positive_words.add(word)
        if sentiment == -1:
            negative_words.add(word)
    return positive_words, negative_words

def load_zhiwang():
    input_file_1 = "D:/datasets/lexicon/sentiment/english_positive.txt"
    fr = open(input_file_1, 'r', encoding='utf-8', errors="ignore")
    positive_words = set()
    for line in fr.readlines():
        if len(line.strip().split(" ")) > 1:
            continue
        positive_words.add(line.strip())
    fr.close()
    input_file_2 = "D:/datasets/lexicon/sentiment/english_negative.txt"
    fr = open(input_file_2, 'r', encoding='utf-8', errors="ignore")
    negative_words = set()
    for line in fr.readlines():
        if len(line.strip().split(" ")) > 1:
            continue
        negative_words.add(line.strip())
    jiaoji = positive_words & negative_words
    positive_words = positive_words - jiaoji
    negative_words = negative_words - jiaoji
    positive = ['proudly', 'ardently', 'blatant', 'flatly', 'solemn', 'dispassionately', 'fantastically', 'delicate',
                'cool', 'quiet', 'fantastic', "cheap"]
    negative = ['costly','painstaking', 'serious', 'chesty', 'seriously']
    for word in positive:
        positive_words.add(word)
    for word in negative:
        negative_words.add(word)
    output_file = open("../data/lexicon_zhiwang.csv", "w", encoding="utf-8")
    for word in positive_words:
        output_file.write("{},1\n".format(word))
    for word in negative_words:
        output_file.write("{},-1\n".format(word))
    output_file.close()
    return positive_words, negative_words


if __name__ == '__main__':
    positive_words_1, negative_words_1 = load_lexicon()
    positive_words_2, negative_words_2 = load_zhiwang()
    # print(positive_words_2)
    # print(len(positive_words_2))
    # bingji = positive_words_1 & positive_words_2
    # print("nice" in bingji, len(bingji))
    # processing_lexicon_NRC(input_path="D:/datasets/lexicon/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", output_path="../data/lexicon.csv")