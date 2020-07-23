import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time

n_samples = 300
n_components = 2
perplexity = 100


def draw(source_dataset, target_dataset, fix_bert=False, methods="bert_PretrainBERT_data_all_val_acc_0.9294_0.6922_Rating_Mask_Multidomain"):
    # fix_bert = False
    # methods = "bert_PretrainBERT_data_all_val_acc_0.9294_0.6922_Rating_Mask_Multidomain"
    method_name = ""
    if methods.find("Rating_Mask_Multidomain") > -1:
        if fix_bert:
            method_name = r'SentimentBERT$_{F}$'
            method_name_ = 'SentimentBERT_F'
        else:
            method_name = r'SentimentBERT'
            method_name_ = 'SentimentBERT'
    else:
        if fix_bert:
            method_name = r'BERT$_{F}$'
            method_name_ = 'BERT_F'
        else:
            method_name = r'BERT'
            method_name_ = 'BERT'

    source_data_path = "../features/{}_{}_{}.npy".format(source_dataset, fix_bert, methods)
    source_data = np.load(source_data_path)
    X_source, y_source = source_data[:, :-1], source_data[:, -1]
    red_source = y_source == 0
    green_source = y_source == 1

    target_data_path = "../features/{}_{}_{}_{}.npy".format(source_dataset, target_dataset, fix_bert, methods)
    target_data = np.load(target_data_path)
    X_target, y_target = target_data[:, :-1], target_data[:, -1]
    # X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    print(X_target.shape, y_target)
    red_target = y_target == 0
    green_target = y_target == 1

    fig = plt.figure()
    ax = fig.add_subplot(111)

    t0 = time()
    tsne_target = manifold.TSNE(n_components=n_components, init='random',
                         random_state=0, perplexity=perplexity)
    Y_target = tsne_target.fit_transform(X_target)

    tsne_source = manifold.TSNE(n_components=n_components, init='random',
                                random_state=0, perplexity=perplexity)
    Y_source = tsne_source.fit_transform(X_source)

    t1 = time()
    print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
    ax.set_title(method_name, fontdict={'fontsize': 15})
    type3 = ax.scatter(Y_target[red_target, 0], Y_target[red_target, 1], c="r", alpha=1, s=1, marker='s')
    type4 = ax.scatter(Y_target[green_target, 0], Y_target[green_target, 1], c="g", alpha=1, s=1, marker='s')
    type1 = ax.scatter(Y_source[red_source, 0], Y_source[red_source, 1], c="blue", alpha=1, s=1, marker='s')
    type2 = ax.scatter(Y_source[green_source, 0], Y_source[green_source, 1], c="m", alpha=1, s=1, marker='s')
    ax.legend((type1, type2, type3, type4), ("Source negative", "Source positive", "Target negative", "Target positive"), loc=0, prop={'size': 11}, markerscale=8, frameon=False, handletextpad=0.5)

    ax.axis('tight')
    plt.savefig("../features/{}_{}_{}_{}.pdf".format(source_dataset, target_dataset, method_name_, perplexity), bbox_inches="tight")
    plt.close()


def draw_2(source_dataset, target_dataset, fix_bert=False, methods="bert_PretrainBERT_data_all_val_acc_0.9294_0.6922_Rating_Mask_Multidomain"):
    # fix_bert = False
    # methods = "bert_PretrainBERT_data_all_val_acc_0.9294_0.6922_Rating_Mask_Multidomain"
    method_name = ""
    if methods.find("Rating_Mask_Multidomain") > -1:
        if fix_bert:
            method_name = r'SentimentBERT$_{F}$'
            method_name_ = 'SentimentBERT_F'
        else:
            method_name = r'SentimentBERT'
            method_name_ = 'SentimentBERT'
    else:
        if fix_bert:
            method_name = r'BERT$_{F}$'
            method_name_ = 'BERT_F'
        else:
            method_name = r'BERT'
            method_name_ = 'BERT'

    source_data_path = "../features/{}_{}_{}.npy".format(source_dataset, fix_bert, methods)
    source_data = np.load(source_data_path)
    X_source, y_source = source_data[:, :-1], source_data[:, -1]
    red_source = y_source == 0
    green_source = y_source == 1

    target_data_path = "../features/{}_{}_{}_{}.npy".format(source_dataset, target_dataset, fix_bert, methods)
    target_data = np.load(target_data_path)
    X_target, y_target = target_data[:, :-1], target_data[:, -1]
    # X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    print(X_target.shape, y_target)
    red_target = y_target == 0
    green_target = y_target == 1

    fig = plt.figure(figsize=(10, 4))
    ax_source = fig.add_subplot(121)
    ax_target = fig.add_subplot(122)
    t0 = time()
    tsne_target = manifold.TSNE(n_components=n_components, init='random',
                         random_state=0, perplexity=perplexity)
    Y_target = tsne_target.fit_transform(X_target)

    tsne_source = manifold.TSNE(n_components=n_components, init='random',
                                random_state=0, perplexity=perplexity)
    Y_source = tsne_source.fit_transform(X_source)

    t1 = time()
    print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
    ax_source.set_title("source", fontdict={'fontsize': 20})
    ax_target.set_title("target", fontdict={'fontsize': 20})
    type3 = ax_target.scatter(Y_target[red_target, 0], Y_target[red_target, 1], c="r", alpha=1, s=1, marker='s')
    type4 = ax_target.scatter(Y_target[green_target, 0], Y_target[green_target, 1], c="blue", alpha=1, s=1, marker='s')
    type1 = ax_source.scatter(Y_source[red_source, 0], Y_source[red_source, 1], c="g", alpha=1, s=1, marker='s')
    type2 = ax_source.scatter(Y_source[green_source, 0], Y_source[green_source, 1], c="m", alpha=1, s=1, marker='s')
    ax_target.legend((type3, type4), ("negative", "positive"), loc=0, prop={'size': 11}, markerscale=8, frameon=False, handletextpad=0.5)
    ax_source.legend((type1, type2), ("negative", "positive"), loc=0, prop={'size': 11}, markerscale=8, frameon=False, handletextpad=0.5)
    ax_source.axis('tight')
    ax_target.axis('tight')
    # plt.title(method_name, fontdict={'fontsize': 15})
    plt.savefig("../features/{}_{}_{}_{}_2.pdf".format(source_dataset, target_dataset, method_name_, perplexity), bbox_inches="tight")
    plt.close()


num = 0
for source_dataset in ['books', 'dvd', 'electronics', 'kitchen_&_housewares']:
    for target_dataset in ['books', 'dvd', 'electronics', 'kitchen_&_housewares']:
        if source_dataset == target_dataset:
            continue
        if source_dataset != "books" or target_dataset != "electronics":
            continue
        for fix_bert in [True, False]:
            for methods in ["bert_PretrainBERT_data_all_val_acc_0.9294_0.6922_Rating_Mask_Multidomain", "gruntdatazhoujiebert_model"]:
                # if num > 0:
                #     continue
                # draw(source_dataset, target_dataset, fix_bert, methods)
                draw_2(source_dataset, target_dataset, fix_bert, methods)
                num += 1


