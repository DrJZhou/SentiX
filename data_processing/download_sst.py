"""This module defines a configurable SSTDataset class."""

import pytreebank
import torch
from torch.utils.data import Dataset

sst = pytreebank.load_sst()


def rpad(array, n=70):
    """Right padding."""
    current_len = len(array)
    if current_len > n:
        return array[: n - 1]
    extra = n - current_len
    return array + ([0] * extra)


def get_binary_label(label):
    """Convert fine-grained label to binary label."""
    if label < 2:
        return 0
    if label > 2:
        return 1
    raise ValueError("Invalid label")


class SSTDataset(Dataset):
    """Configurable SST Dataset.

    Things we can configure:
        - split (train / val / test)
        - root / all nodes
        - binary / fine-grained
    """

    def __init__(self, split="train", root=True, binary=True):
        """Initializes the dataset with given configuration.
        Args:
            split: str
                Dataset split, one of [train, val, test]
            root: bool
                If true, only use root nodes. Else, use all nodes.
            binary: bool
                If true, use binary labels. Else, use fine-grained.
        """
        self.sst = sst[split]

        if root and binary:
            self.data = [
                (
                    tree.to_lines()[0].lower(),
                    get_binary_label(tree.label),
                )
                for tree in self.sst
                if tree.label != 2
            ]
        elif root and not binary:
            self.data = [
                (
                    tree.to_lines()[0].lower(),
                    tree.label,
                )
                for tree in self.sst
            ]
        elif not root and not binary:
            self.data = [
                (line.lower(), label)
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
            ]
        else:
            self.data = [
                (
                    line.lower(),
                    get_binary_label(label),
                )
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
                if label != 2
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, y = self.data[index]
        X = torch.tensor(X)
        return X, y


import os
import numpy as np

if __name__ == '__main__':
    for root in [True, False]:
        for binary in [True, False]:
            class_num = 5
            class_name = "all"
            if root:
                class_name = 'root'
            if binary:
                class_num = 2
            path_to = '../data/sentiment_analysis/SST-{}-{}/'.format(class_num, class_name)
            if not os.path.exists(path_to):
                os.mkdir(path_to)
            trainset = SSTDataset("train", root=root, binary=binary)
            devset = SSTDataset("dev", root=root, binary=binary)
            testset = SSTDataset("test", root=root, binary=binary)
            fr_to_train = open(path_to+"train.txt", 'w', encoding='utf-8')
            fr_to_val = open(path_to + "val.txt", 'w', encoding='utf-8')
            fr_to_test = open(path_to + "test.txt", 'w', encoding='utf-8')
            np.random.shuffle(trainset.data)
            for data in trainset.data:
                fr_to_train.write("{}\t{}\n".format(data[0], data[1]))
            fr_to_train.close()
            np.random.shuffle(devset.data)
            for data in devset.data:
                fr_to_val.write("{}\t{}\n".format(data[0], data[1]))
            fr_to_val.close()
            np.random.shuffle(testset.data)
            for data in testset.data:
                fr_to_test.write("{}\t{}\n".format(data[0], data[1]))
            fr_to_test.close()