import os


def statistic_aspect_based_sentiment_analysis():
    base_path = 'results/aspect_based_sentiment_analysis/'
    fname = os.listdir(base_path)
    ans = []
    for name in fname:
        fr = open(base_path+name, 'r')
        dataset, bert_path, fix_bert, test_acc = None, None, None, None
        for line in fr.readlines():
            line = line.strip()
            if line.startswith(">>> dataset: "):
                dataset = line.replace(">>> dataset: ", "")
            if line.startswith(">>> pretrained_bert_name: "):
                bert_path = line.replace(">>> pretrained_bert_name: ", "")
            if line.startswith(">>> fix_bert: "):
                fix_bert = line.replace(">>> fix_bert: ", "")
            if line.startswith(">> test_acc: "):
                test_acc = line.replace(">> test_acc: ", "").split(", test_f1: ")[0]
        if test_acc:
            ans.append(["aspect_based_sentiment_analysis", dataset, bert_path, fix_bert, test_acc])
    return ans


def statistic_sentiment_analysis():
    base_path = 'results/sentiment_analysis/'
    fname = os.listdir(base_path)
    ans = []
    for name in fname:
        fr = open(base_path + name, 'r')
        dataset, bert_path, fix_bert, test_acc = None, None, None, None
        for line in fr.readlines():
            line = line.strip()
            if line.startswith(">>> dataset: "):
                dataset = line.replace(">>> dataset: ", "")
            if line.startswith(">>> bert_path: "):
                bert_path = line.replace(">>> bert_path: ", "")
            if line.startswith(">>> fix_bert: "):
                fix_bert = line.replace(">>> fix_bert: ", "")
            if line.startswith(">> test_acc: "):
                test_acc = line.replace(">> test_acc: ", "").split(", test_f1: ")[0]
        if test_acc:
            ans.append(["sentiment_analysis", dataset, bert_path, fix_bert, test_acc])
    return ans


def statistic_few_shot_sentiment_analysis():
    base_path = 'results/few_shot_sentiment_analysis/'
    fname = os.listdir(base_path)
    ans = []
    for name in fname:
        fr = open(base_path + name, 'r')
        dataset, bert_path, fix_bert, few_shot_num, test_acc = None, None, None, None, None
        for line in fr.readlines():
            line = line.strip()
            if line.startswith(">>> dataset: "):
                dataset = line.replace(">>> dataset: ", "")
            if line.startswith(">>> bert_path: "):
                bert_path = line.replace(">>> bert_path: ", "")
            if line.startswith(">>> fix_bert: "):
                fix_bert = line.replace(">>> fix_bert: ", "")
            if line.startswith(">>> few_shot_num: "):
                few_shot_num = line.replace(">>> few_shot_num: ", "")
            if line.startswith(">> test_acc: "):
                test_acc = line.replace(">> test_acc: ", "").split(", test_f1: ")[0]
        if test_acc:
            ans.append(["few_shot_sentiment_analysis", dataset, bert_path, fix_bert, few_shot_num, test_acc])
    return ans


def statistic_multidomain_sentiment_analysis():
    base_path = 'results/multidomain/'
    fname = os.listdir(base_path)
    ans = []
    for name in fname:
        fr = open(base_path + name, 'r')
        dataset, bert_path, fix_bert, test_acc = None, None, None, None
        test_acc_books, test_acc_dvd, test_acc_electronics, test_acc_kitchen_housewares = None, None, None, None
        for line in fr.readlines():
            line = line.strip()
            if line.startswith(">>> dataset: "):
                dataset = line.replace(">>> dataset: ", "")
            if line.startswith(">>> bert_path: "):
                bert_path = line.replace(">>> bert_path: ", "")
            if line.startswith(">>> fix_bert: "):
                fix_bert = line.replace(">>> fix_bert: ", "")
            if line.startswith(">> name: books, test_acc: "):
                test_acc_books = line.replace(">> name: books, test_acc: ", "").split(", test_f1: ")[0]
            if line.startswith(">> name: dvd, test_acc: "):
                test_acc_dvd = line.replace(">> name: dvd, test_acc: ", "").split(", test_f1: ")[0]
            if line.startswith(">> name: electronics, test_acc: "):
                test_acc_electronics = line.replace(">> name: electronics, test_acc: ", "").split(", test_f1: ")[0]
            if line.startswith(">> name: kitchen_&_housewares, test_acc: "):
                test_acc_kitchen_housewares = line.replace(">> name: kitchen_&_housewares, test_acc: ", "").split(", test_f1: ")[0]
        if test_acc_books or test_acc_electronics or test_acc_dvd or test_acc_kitchen_housewares:
            if not test_acc_books:
                test_acc_books = "0"
            if not test_acc_dvd:
                test_acc_dvd = "0"
            if not test_acc_electronics:
                test_acc_electronics = "0"
            if not test_acc_kitchen_housewares:
                test_acc_kitchen_housewares = "0"
            ans.append(["multidomain_sentiment_analysis", dataset, bert_path, fix_bert, test_acc_books, test_acc_dvd, test_acc_electronics, test_acc_kitchen_housewares])
    return ans


def statistic_few_shot_multidomain_sentiment_analysis():
    base_path = 'results/few_shot_multidomain/'
    fname = os.listdir(base_path)
    ans = []
    for name in fname:
        fr = open(base_path + name, 'r')
        dataset, bert_path, fix_bert, test_acc = None, None, None, None
        few_shot_rate = None
        test_acc_books, test_acc_dvd, test_acc_electronics, test_acc_kitchen_housewares = None, None, None, None
        for line in fr.readlines():
            line = line.strip()
            if line.startswith(">>> dataset: "):
                dataset = line.replace(">>> dataset: ", "")
            if line.startswith(">>> bert_path: "):
                bert_path = line.replace(">>> bert_path: ", "")
            if line.startswith(">>> few_shot_rate: "):
                few_shot_rate = line.replace(">>> few_shot_rate: ", "")
            if line.startswith(">>> fix_bert: "):
                fix_bert = line.replace(">>> fix_bert: ", "")
            if line.startswith(">> name: books, test_acc: "):
                test_acc_books = line.replace(">> name: books, test_acc: ", "").split(", test_f1: ")[0]
            if line.startswith(">> name: dvd, test_acc: "):
                test_acc_dvd = line.replace(">> name: dvd, test_acc: ", "").split(", test_f1: ")[0]
            if line.startswith(">> name: electronics, test_acc: "):
                test_acc_electronics = line.replace(">> name: electronics, test_acc: ", "").split(", test_f1: ")[0]
            if line.startswith(">> name: kitchen_&_housewares, test_acc: "):
                test_acc_kitchen_housewares = line.replace(">> name: kitchen_&_housewares, test_acc: ", "").split(", test_f1: ")[0]
        if test_acc_books or test_acc_electronics or test_acc_dvd or test_acc_kitchen_housewares:
            if not test_acc_books:
                test_acc_books = "0"
            if not test_acc_dvd:
                test_acc_dvd = "0"
            if not test_acc_electronics:
                test_acc_electronics = "0"
            if not test_acc_kitchen_housewares:
                test_acc_kitchen_housewares = "0"
            ans.append(["few_shot_multidomain_sentiment_analysis", dataset, bert_path, fix_bert, test_acc_books, test_acc_dvd, test_acc_electronics, test_acc_kitchen_housewares, few_shot_rate])
    return ans


if __name__ == '__main__':
    ans_aspect_based_sentiment_analysis = statistic_aspect_based_sentiment_analysis()
    ans_sentiment_analysis = statistic_sentiment_analysis()
    ans_few_shot_sentiment_analysis = statistic_few_shot_sentiment_analysis()
    ans_multidomain_sentiment_analysis = statistic_multidomain_sentiment_analysis()
    ans_few_shot_multidomain_sentiment_analysis = statistic_few_shot_multidomain_sentiment_analysis()
    fr_to = open("results.csv", 'w')
    for data in ans_aspect_based_sentiment_analysis:
        fr_to.write(",".join(data)+"\n")

    for data in ans_sentiment_analysis:
        fr_to.write(",".join(data)+"\n")

    for data in ans_few_shot_sentiment_analysis:
        fr_to.write(",".join(data) + "\n")

    for data in ans_multidomain_sentiment_analysis:
        fr_to.write(",".join(data) + "\n")

    for data in ans_few_shot_multidomain_sentiment_analysis:
        fr_to.write(",".join(data) + "\n")

    fr_to.close()