'''
multi-domain sentiment classification

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
from torch.utils.data import DataLoader, random_split, ConcatDataset
from data_utils_multidomain import Tokenizer4Bert, PreTrainDataset
from models.SentimentClassification import SentimentClassification
# try:
#     from apex import amp
# except ImportError:
#     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler(sys.stdout))
bert_path = "/gruntdata/zhoujie/bert_model"
# bert_path = 'state_dict/bert_PretrainBERT_data_all_val_f1_0.2305_acc_0.6389_Rating'
test_flag=False


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        if self.opt.bert_path:
            bert_path = self.opt.bert_path
        bert = BertModel.from_pretrained(bert_path)
        self.model = opt.model_class(bert, opt).to(opt.device)
        self.trainset = PreTrainDataset(opt.dataset, tokenizer)
        np.random.shuffle(self.trainset.data)
        self.testsets = []
        for name in ['books', 'dvd', 'electronics', 'kitchen_&_housewares']:
            if name != opt.dataset:
                self.testsets.append([name, PreTrainDataset(name, tokenizer)])
        # np.random.shuffle(self.testset.data)
        import copy
        self.valset = copy.deepcopy(self.trainset)
        train_num = int(len(self.trainset.data)*0.9)
        self.trainset.data = self.trainset.data[: train_num]
        self.valset = self.valset.data[train_num:]

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)


    def _inference(self, data_loader):
        representations = []
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['sentiment'].numpy()
                _, t_outputs = self.model.bert(t_inputs[0])
                t_outputs = t_outputs.cpu().numpy()
                for i in range(len(t_targets)):
                    representations.append(np.concatenate((t_outputs[i], [t_targets[i]])))
        return np.array(representations)

    def run(self):
        # Loss and Optimizer
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=False)
        # val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
        best_model_path = 'state_dict/{0}_{1}_val_temp_{2}_{3}.pkl'.format(self.opt.model_name, self.opt.dataset,
                                                                       self.opt.fix_bert, self.opt.bert_path.replace("state_dict/", "").replace("/", ""))
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        train_representation = self._inference(train_data_loader)
        train_path = "features/{}_{}_{}.npy".format(self.opt.dataset, self.opt.fix_bert,
                                                    self.opt.bert_path.replace("state_dict/", "").replace("/", ""))
        np.save(train_path, train_representation)
        for i in range(len(self.testsets)):
            test_data_loader = DataLoader(dataset=self.testsets[i][1], batch_size=self.opt.batch_size, shuffle=False)
            test_representation = self._inference(test_data_loader)
            test_path = "features/{}_{}_{}_{}.npy".format(self.opt.dataset, self.testsets[i][0], self.opt.fix_bert,
                                                        self.opt.bert_path.replace("state_dict/", "").replace("/", ""))
            np.save(test_path, test_representation)


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='sentiment_classification', type=str)
    parser.add_argument('--dataset', default='sst-5', type=str, help='sst-2')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=15, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=256, type=int)
    parser.add_argument('--sentiment_class', default=2, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    parser.add_argument("--circle_loss", action="store_true", help="Whether to use circle loss")
    parser.add_argument("--fix_bert", action="store_true", help="Whether to fix bert")
    parser.add_argument('--cross_val_fold', default=-1, type=int, help='k-fold cross validation')
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument('--bert_path', default='/gruntdata/zhoujie/bert_model', type=str)
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'sentiment_classification': SentimentClassification,
    }

    input_colses = {
        'sentiment_classification': ['text_bert_indices'],
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

    log_file = 'results/multidomain/{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()