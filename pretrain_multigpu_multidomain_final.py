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
from data_utils_pretrain_multidomain_final import Tokenizer4Bert, SentimentTokenizer, EmojiTokenizer, PreTrainDataset
from models.PretrainBERT_rating_mask_final import PretrainBERT
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
# try:
#     from apex import amp
# except ImportError:
#     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler(sys.stdout))
bert_path = "/gruntdata/zhoujie/bert_model"
test_flag = False


class Instructor:
    def __init__(self, opt):
        emoji_tokenizer = EmojiTokenizer()
        sentiment_tokenizer = SentimentTokenizer()
        opt.emoji_class = emoji_tokenizer.emoji_class
        opt.sentiment_class = sentiment_tokenizer.sentiment_class
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(bert_path)
        self.model = opt.model_class(bert, opt).to(opt.device)
        dataset = PreTrainDataset(opt.dataset, tokenizer, emoji_tokenizer, sentiment_tokenizer)
        np.random.shuffle(dataset.data)
        train_num = int(len(dataset.data)*0.999)
        test_num = len(dataset.data) - train_num
        self.trainset, self.testset = random_split(dataset, (train_num, test_num))
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

    def circle_loss(self, y_pred, y_true):
        has_pos_sample = torch.sum(y_true, dim=-1)!=0
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1]).to(device=self.opt.device) + 0.5
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.zeros(y_true.size(0)).to(device=self.opt.device)
        pos_loss = torch.zeros(y_true.size(0)).to(device=self.opt.device)
        neg_loss[has_pos_sample] = torch.logsumexp(y_pred_neg[has_pos_sample], dim=-1).float()
        pos_loss[has_pos_sample] = torch.logsumexp(y_pred_pos[has_pos_sample], dim=-1).float()
        loss = neg_loss + pos_loss
        return torch.mean(loss)

    def _train(self, train_data_loader, val_data_loader):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        max_grad_norm = 1.0
        num_training_steps = len(train_data_loader)*3/self.opt.gradient_accumulation_steps
        num_warmup_steps = 2000
        # for name, param in self.model.bert.named_parameters():
        #     param.requires_grad = False
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = AdamW(_params, lr=self.opt.learning_rate,
                          correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)  # PyTorch scheduler
        # optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        if self.opt.fp16:
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.opt.fp16_opt_level)
        if self.opt.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if self.opt.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.opt.local_rank], output_device=self.opt.local_rank
            )

        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None
        gradient_true = False
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total_word, n_total_sentence, n_total_loss, loss_total = np.array(
                [0.0, 0.0]), 0, [0, 0], 0, 0
            # switch model to training mode
            self.model.train()
            batch_num = len(train_data_loader)
            for i_batch, sample_batched in enumerate(train_data_loader):
                if test_flag and i_batch > 10:
                    break
                # if global_step > self.opt.gradient_accumulation_steps*500 and gradient_true == False:
                #     gradient_true = True
                #     for name, param in self.model.bert.named_parameters():
                #         param.requires_grad = True
                #     _params = filter(lambda p: p.requires_grad, self.model.parameters())
                #     optimizer = AdamW(_params, lr=self.opt.learning_rate,
                #                       correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
                #     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                #                                                 num_training_steps=num_training_steps)
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                max_len = torch.max(torch.sum(inputs[0]!=0, dim=-1))
                inputs[0] = inputs[0][:, :max_len]
                word_labels = sample_batched['word_labels'].to(self.opt.device)
                word_labels = word_labels[:, :max_len].contiguous()
                word_labels = word_labels.view(-1)
                rating = sample_batched['rating'].to(self.opt.device)
                labels = [word_labels, rating]
                outputs = self.model(inputs, labels)

                loss = outputs[-1]
                if self.opt.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if self.opt.gradient_accumulation_steps > 1:
                    loss /= self.opt.gradient_accumulation_steps
                if self.opt.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(_params, max_grad_norm)
                global_step += 1
                if global_step%self.opt.gradient_accumulation_steps == 0:
                    optimizer.step()
                    # clear gradient accumulators
                    optimizer.zero_grad()
                    scheduler.step()

                n_correct[0] += (torch.argmax(outputs[0], -1) == word_labels)[word_labels!=0].sum().item()
                n_total_word += (word_labels != 0).sum().item()
                n_correct[1] += (torch.argmax(outputs[1], -1) == rating).sum().item()

                loss_total += loss.item() * len(inputs[0])
                n_total_loss += len(inputs[0])
                if global_step % self.opt.log_step == 0:
                    train_acc_word = n_correct[0] / n_total_word
                    train_loss = loss_total / n_total_loss
                    train_acc_rating = n_correct[1] / n_total_loss
                    logger.info(
                        '{}/{}---loss: {:.4f}, acc: {:.4f}, {:.4f}'.format(i_batch, batch_num, train_loss, train_acc_word, train_acc_rating))

                if global_step % int(batch_num/20) == 0:
                    # print(i_batch, int(batch_num/20))
                    val_acc = self._evaluate_acc_f1(val_data_loader)
                    logger.info('> val_acc: {:.4f}, {:.4f}'.format(val_acc[0], val_acc[1]))
                    if val_acc[0] > max_val_acc:
                        max_val_acc = val_acc[0]


                    if not os.path.exists('state_dict'):
                        os.mkdir('state_dict')
                    path = 'state_dict/{0}_{1}_val_acc_{2}_{3}_Rating_Mask_Multidomain'.format(self.opt.model_name,
                                                                                         self.opt.dataset,
                                                                                         round(val_acc[0], 4),
                                                                                         round(val_acc[1], 4))
                    try:
                        torch.save(self.model.state_dict(), path)
                    except:
                        torch.save(self.model.module.state_dict(), path)
                    path_bert = 'state_dict/bert_{0}_{1}_val_acc_{2}_{3}_Rating_Mask_Multidomain'.format(self.opt.model_name,
                                                                                                   self.opt.dataset,
                                                                                                   round(val_acc[0], 4),
                                                                                                   round(val_acc[1], 4))
                    if not os.path.exists(path_bert):
                        os.mkdir(path_bert)
                    try:
                        self.model.bert.save_pretrained(path_bert)
                    except:
                        self.model.module.bert.save_pretrained(path_bert)
                    logger.info('>> saved: {}'.format(path))
        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total, n_total_word = np.array([0.0, 0.0]), 0, 0.0
        n_f1, n_total_f1 = np.array([0.0, 0.0]), 0
        t_targets_all, t_outputs_all = None, [None, None]
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                if (test_flag and t_batch > 10) or t_batch > 1000:
                    break
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                max_len = torch.max(torch.sum(t_inputs[0] != 0, dim=-1))
                t_inputs[0] = t_inputs[0][:, :max_len]
                t_word_labels = t_sample_batched['word_labels'].to(self.opt.device)
                t_word_labels = t_word_labels[:, :max_len].contiguous()
                t_word_labels = t_word_labels.view(-1)
                t_rating = t_sample_batched['rating'].to(self.opt.device)
                labels = [t_word_labels, t_rating]
                t_outputs = self.model(t_inputs)
                t_outputs = list(t_outputs)

                t_outputs[0] = t_outputs[0][t_word_labels != 0]
                labels[0] = labels[0][t_word_labels != 0]
                n_correct[0] += (torch.argmax(t_outputs[0], -1) == labels[0]).sum().item()
                n_correct[1] += (torch.argmax(t_outputs[1], -1) == labels[1]).sum().item()
                n_total_word += len(t_outputs[0])
                n_total += len(t_outputs[1])
                # for i in range(len(labels)):
                #     labels[i] = labels[i].cpu()
                #
                # for i in range(len(t_outputs)):
                #     t_outputs[i] = t_outputs[i].cpu()
                #
                # if t_targets_all is None:
                #     t_targets_all = labels
                #     for i in range(len(t_outputs)):
                #         t_outputs_all[i] = t_outputs[i]
                # else:
                #     for i in range(len(t_outputs)):
                #         t_targets_all[i] = torch.cat((t_targets_all[i], labels[i]), dim=0)
                #         t_outputs_all[i] = torch.cat((t_outputs_all[i], t_outputs[i]), dim=0)
                #
                # if t_batch % 50 == 0:
                #     for i in range(len(t_outputs_all)):
                #         if i == 0:
                #             n_f1[i] += metrics.f1_score(t_targets_all[i].view(-1),
                #                                    torch.argmax(t_outputs_all[i], -1).view(-1), average='macro')
                #         if i == 1:
                #             n_f1[i] += metrics.f1_score(t_targets_all[i], torch.argmax(t_outputs_all[i], -1), labels=[0, 1, 2, 3, 4], average='macro')
                #     t_targets_all, t_outputs_all = None, [None, None]
                #     n_total_f1 += 1
        acc = [0.0, 0.0]
        acc[0] = n_correct[0] / n_total_word
        acc[1] = n_correct[1] / n_total
        # f1 = n_f1 / n_total_f1
        return acc

    def run(self):
        # Loss and Optimizer

        self.opt.train_batch_size = self.opt.batch_size * max(1, self.opt.n_gpu)
        train_sampler = RandomSampler(self.trainset) if self.opt.local_rank == -1 else DistributedSampler(self.trainset)
        train_data_loader = DataLoader(dataset=self.trainset, sampler=train_sampler,
                                       batch_size=self.opt.train_batch_size, num_workers=8)
        test_sampler = SequentialSampler(self.testset)
        test_data_loader = DataLoader(dataset=self.testset, sampler=test_sampler, batch_size=self.opt.batch_size,
                                      num_workers=8)

        self._reset_params()

        # best_model_path = "state_dict/PretrainBERT_data_all_val_acc_0.9319_0.0065_Rating_Mask_415280"
        # best_model = torch.load(best_model_path)
        # rename_best_model = {}
        # for name in best_model:
        #     rename_best_model[name.replace("module.", "")] = best_model[name]
        # self.model.load_state_dict(rename_best_model)
        # # self.model = self.model.module
        best_model_path = self._train(train_data_loader, test_data_loader)
        # self.model.load_state_dict(torch.load(best_model_path))
        # self.model.eval()
        # test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        # logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


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
    parser.add_argument('--log_step', default=128, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument("--circle_loss", action="store_true", help="Whether to use circle loss")
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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="")
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
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
    #python pretrain_multigpu_final.py --dataset data_all --batch_size 16 --max_seq_len 256