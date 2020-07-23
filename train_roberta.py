'''
sentiment classification

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

from transformers import BertModel, RobertaModel

from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from data_utils_roberta import Tokenizer4Bert, PreTrainDataset
from models.SentimentClassification import SentimentClassification

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler(sys.stdout))
bert_path = "/gruntdata/zhoujie/roberta_model"
# bert_path = 'state_dict/bert_PretrainBERT_data_all_val_f1_0.2305_acc_0.6389_Rating'
test_flag=False


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        if self.opt.bert_path:
            bert_path = self.opt.bert_path.replace("\r", "").replace("\n", "")
        bert = RobertaModel.from_pretrained(bert_path)
        self.model = opt.model_class(bert, opt).to(opt.device)
        self.trainset = PreTrainDataset(opt.dataset, tokenizer, train_or_test='train')
        np.random.shuffle(self.trainset.data)
        self.testset = PreTrainDataset(opt.dataset, tokenizer, train_or_test='test')
        # np.random.shuffle(self.testset.data)
        if self.opt.cross_val_fold < 0:
            self.valset = PreTrainDataset(opt.dataset, tokenizer, train_or_test='val')
        if self.opt.cross_val_fold == 0:
            self.valset = self.testset

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

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None
        for epoch in range(self.opt.num_epoch):
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            if self.opt.fix_bert:
                for name, param in self.model.bert.named_parameters():
                    param.requires_grad = False
                _params = filter(lambda p: p.requires_grad, self.model.parameters())
                optimizer = self.opt.optimizer(_params, lr=0.0001, weight_decay=self.opt.l2reg)
            else:
                if epoch < 1:
                    for name, param in self.model.bert.named_parameters():
                        param.requires_grad = False
                    _params = filter(lambda p: p.requires_grad, self.model.parameters())
                    optimizer = self.opt.optimizer(_params, lr=0.0001, weight_decay=self.opt.l2reg)
                else:
                    for name, param in self.model.bert.named_parameters():
                        param.requires_grad = True
                    _params = filter(lambda p: p.requires_grad, self.model.parameters())
                    optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['sentiment'].to(self.opt.device)

                loss = criterion(outputs, targets)
                if self.opt.gradient_accumulation_steps > 1:
                    loss /= self.opt.gradient_accumulation_steps
                    torch.nn.utils.clip_grad_norm_(_params, 1.0)
                loss.backward()
                if global_step%self.opt.gradient_accumulation_steps == 0:
                    optimizer.step()
                    # clear gradient accumulators
                    optimizer.zero_grad()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_temp_roberta'.format(self.opt.model_name, self.opt.dataset)
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['sentiment'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        test_data_loader = DataLoader(dataset=self.testset, batch_size=16, shuffle=False)
        if self.opt.cross_val_fold > 0:
            valset_len = len(self.trainset) // self.opt.cross_val_fold
            splitedsets = random_split(self.trainset, tuple([valset_len] * (self.opt.cross_val_fold - 1) + [len(self.trainset) - valset_len * (self.opt.cross_val_fold - 1)]))

            all_test_acc, all_test_f1 = [], []
            for fid in range(self.opt.cross_val_fold):
                logger.info('fold : {}'.format(fid))
                logger.info('>' * 100)
                trainset = ConcatDataset([x for i, x in enumerate(splitedsets) if i != fid])
                valset = splitedsets[fid]
                train_data_loader = DataLoader(dataset=trainset, batch_size=self.opt.batch_size, shuffle=True)
                val_data_loader = DataLoader(dataset=valset, batch_size=16, shuffle=False)

                self._reset_params()
                best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)

                self.model.load_state_dict(torch.load(best_model_path))
                test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
                all_test_acc.append(test_acc)
                all_test_f1.append(test_f1)
                logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))

            mean_test_acc, mean_test_f1 = numpy.mean(all_test_acc), numpy.mean(all_test_f1)
            logger.info('>' * 100)
            logger.info('>>> mean_test_acc: {:.4f}, mean_test_f1: {:.4f}'.format(mean_test_acc, mean_test_f1))
        else:
            train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
            test_data_loader = DataLoader(dataset=self.testset, batch_size=16, shuffle=False)
            val_data_loader = DataLoader(dataset=self.valset, batch_size=16, shuffle=False)

            self._reset_params()
            best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
            self.model.load_state_dict(torch.load(best_model_path))
            self.model.eval()
            test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
            logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='sentiment_classification', type=str)
    parser.add_argument('--dataset', default='sst-5', type=str, help='sst-2')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=20, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=4, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--log_step', default=64, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=256, type=int)
    parser.add_argument('--sentiment_class', default=5, type=int)
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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="")
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

    log_file = 'results/sentiment_analysis/{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()