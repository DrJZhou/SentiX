'''
help to deal with the dataset
'''

from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer
import re
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
import string
tknzr = TweetTokenizer()


# regular rule of emoticon
EMOTICONS = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      |
      <3                         # heart
    )"""
EMOTICON_RE = re.compile(EMOTICONS, re.VERBOSE | re.I | re.UNICODE)


# tokenize, del url and so on
def text_processing(text):
    text = " ".join(tknzr.tokenize(text))
    text = text.replace("\r", " ").replace("\n", " ")
    tmp = []
    for w in text.strip().split(" "):
        if len(w.strip())==0:
            continue
        if not judge_url(w):
            if judge_emoji(w):
                tmp.append(w)
            else:
                tmp.append(w.lower())
    return tmp


# judge if a word is an emoji
def judge_emoji(word):
    if EMOTICON_RE.search(word):
        return True
    return False


# judge if a word is url
def judge_url(word):
    if re.match(r'^https?:/{2}\w.+$', word):
        return True
    else:
        return False


# using sentiwordnet to calculate the score of word
def word_sentiment_score(text_token):
    ttt = nltk.pos_tag(text_token)
    word_tag_fq = nltk.FreqDist(ttt)
    wordlist = word_tag_fq.most_common()

    key = []
    part = []
    frequency = []
    for i in range(len(wordlist)):
        key.append(wordlist[i][0][0])
        part.append(wordlist[i][0][1])
        frequency.append(wordlist[i][1])
    textdf = pd.DataFrame({'key': key,
                           'part': part,
                           'frequency': frequency},
                          columns=['key', 'part', 'frequency'])

    pos_tag = {
        "NN": "n", "NNP": "n", "NNPS": "n", "NNS": "n", "UH": "n",
        'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v',
        'JJ': 'a', 'JJR': 'a', 'JJS': 'a',
        'RB': 'r', 'RBR': 'r', 'RBS': 'r', 'RP': 'r', 'WRB': 'r'
    }

    for i in range(len(textdf['key'])):
        z = textdf.iloc[i, 1]
        textdf.iloc[i, 1] = pos_tag.get(z, '')

    score = {}
    for i in range(len(textdf['key'])):
        m = list(swn.senti_synsets(textdf.iloc[i, 0], textdf.iloc[i, 1]))
        s = 0
        ra = 0
        if len(m) > 0:
            for j in range(len(m)):
                s += (m[j].pos_score() - m[j].neg_score()) / (j + 1)
                ra += 1 / (j + 1)
            score[textdf.iloc[i, 0]] = s / ra
        else:
            score[textdf.iloc[i, 0]] = 0
    # print(score)
    return score


# load the lexicon
def load_lexicon():
    fr = open("../data/sentiment_lexicon_all.csv", 'r', encoding='utf-8')
    lexicon_set = {}
    for line in fr.readlines():
        data = line.strip().split(",")
        word = data[0]
        sentiment = int(data[-1])
        lexicon_set[word] = sentiment
    return lexicon_set


# load the lexicon
sentiment_lexicon = load_lexicon()


# judge the sentiment of word (0: negative, 1: neutral, 2: positive) according to sentiment lexicon and sentiwordnet
def judge_sentiment(text_token):
    # score = word_sentiment_score(text_token)
    sentiment_labels = []
    for i in range(len(text_token)):
        word = text_token[i]
        # word_score = score[word]
        if word in sentiment_lexicon:
            if sentiment_lexicon[word] == 1:
                sentiment_labels.append(2)
            elif sentiment_lexicon[word] == -1:
                sentiment_labels.append(0)
            else:
                sentiment_labels.append(1)
        else:
            sentiment_labels.append(1)
    assert len(text_token) == len(sentiment_labels)
    return sentiment_labels


# load top_k emoji
def load_top_k_emoji(top_k=25):
    fr = open("../data/emoticon_all.csv", "r", encoding='utf-8')
    emoticons = {}
    for line in fr.readlines()[:top_k]:
        data = line.strip().split(",")
        emoji = ",".join(data[:-1])
        emoticons[emoji] = 1
    return emoticons


emoticons_all = load_top_k_emoji(top_k=25)


# judge if word in top_k emoji for the text token
def judge_emotion(text_token):
    emoji_label = []
    for word in text_token:
        if word in emoticons_all:
            emoji_label.append(1)
        else:
            emoji_label.append(0)
    return emoji_label


# print(judge_sentiment(text_processing("It's a nice day!")))



