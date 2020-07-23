'''
Only predict the rating and word-level mask, all the word will be calculated into loss
'''
import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy
import numpy as np

from transformers import BertModel

from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from data_utils import Tokenizer4Bert, PreTrainDataset
from models.PretrainBERT_rating_mask import PretrainBERT

# try:
#     from apex import amp
# except ImportError:
#     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler(sys.stdout))
bert_path = "/gruntdata/zhoujie/bert_model"
test_flag = False


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(bert_path)
        self.model = opt.model_class(bert, opt).to(opt.device)
        self.testset = PreTrainDataset(opt.dataset, tokenizer, train_or_test='test')
        if opt.device.type == 'cuda':
            print('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0.0, 0
        n_f1, n_total_f1 = np.array([0.0, 0.0]), 0
        t_targets_all, t_outputs_all = None, [None, None]
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_rating = t_sample_batched['sentiment'].to(self.opt.device)
                t_outputs = self.model(t_inputs)
                rating_pred = t_outputs[1]
                rating_pred = torch.nn.functional.softmax(rating_pred, dim=-1)
                if self.opt.sentiment_class == 2:
                    rating_pred = (torch.sum(rating_pred[:, 3:], dim=-1) - 0.2 > torch.sum(rating_pred[:, : 2], dim=-1)).long()
                else:
                    rating_pred = torch.argmax(rating_pred, -1)
                print(rating_pred)
                n_correct += (rating_pred == t_rating).sum().item()
                n_total += len(t_rating)

        acc = n_correct / n_total
        return acc

    def run(self):
        test_sampler = SequentialSampler(self.testset)
        test_data_loader = DataLoader(dataset=self.testset, sampler=test_sampler, batch_size=self.opt.batch_size,
                                      num_workers=8)
        best_model_path = "state_dict/PretrainBERT_data_all_val_f1_0.3383_meanf1_0.3906_Rating_Mask"
        if self.opt.best_model_path:
            best_model_path = self.opt.best_model_path
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        test_acc = self._evaluate_acc_f1(test_data_loader)
        print('>> test_acc: {:.4f}'.format(test_acc))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='PretrainBERT', type=str)
    parser.add_argument('--dataset', default='All_Beauty', type=str, help='All_Beauty, AMAZON_FASHION, Yelp')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--best_model_path', default=None, type=str)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--sentiment_class', default=2, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'PretrainBERT': PretrainBERT,
    }

    input_colses = {
        'PretrainBERT': ['text_bert_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    opt.n_gpu = torch.cuda.device_count()

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()