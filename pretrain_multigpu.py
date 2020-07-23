'''
pre-train
Only predict the rating and word-level opinion word, emoji, sentiment mask, only the masked words are calculated into loss
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
from data_utils_pretrain import Tokenizer4Bert, SentimentTokenizer, EmojiTokenizer, PreTrainDataset
from models.PretrainBERT import PretrainBERT

# try:
#     from apex import amp
# except ImportError:
#     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None
        criterion2 = nn.BCELoss()
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total_word, n_total_sentence, n_total_loss, loss_total = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0, [0, 0], 0, 0
            # switch model to training mode
            self.model.train()
            batch_num = len(train_data_loader)
            for i_batch, sample_batched in enumerate(train_data_loader):
                if test_flag and i_batch > 10:
                    break
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                # 'word_sentiment_labels', 'word_sentiment_word_labels', 'word_emoji_labels',
                #                  'sentence_sentiment_labels', 'sentence_emoji_labels'
                outputs = list(outputs)
                text_bert_indices = inputs[0]
                masked_tokens = text_bert_indices == 103
                masked_tokens = masked_tokens.view(-1)
                word_sentiment_labels = sample_batched['word_sentiment_labels'].to(self.opt.device)
                word_sentiment_word_labels = sample_batched['word_sentiment_word_labels'].to(self.opt.device)
                word_emoji_labels = sample_batched['word_emoji_labels'].to(self.opt.device)
                rating = sample_batched['rating'].to(self.opt.device)
                outputs[0] = outputs[0].view(-1, outputs[0].size(2))[masked_tokens]
                outputs[1] = outputs[1].view(-1, outputs[1].size(2))[masked_tokens]
                outputs[2] = outputs[2].view(-1, outputs[2].size(2))[masked_tokens]
                word_sentiment_labels = word_sentiment_labels.view(-1)[masked_tokens]
                word_sentiment_word_labels = word_sentiment_word_labels.view(-1)[masked_tokens]
                word_emoji_labels = word_emoji_labels.view(-1)[masked_tokens]
                loss1 = criterion(outputs[0], word_sentiment_labels)
                loss2 = criterion(outputs[1], word_sentiment_word_labels)
                loss3 = criterion(outputs[2], word_emoji_labels)

                # print(outputs[3].type(), sentence_sentiment_labels.type())
                sentence_sentiment_labels = sample_batched['sentence_sentiment_labels'].to(self.opt.device)
                sentence_emoji_labels = sample_batched['sentence_emoji_labels'].to(self.opt.device)
                sentence_sentiment_labels = sentence_sentiment_labels.float()
                sentence_emoji_labels = sentence_emoji_labels.float()
                if self.opt.circle_loss:
                    loss4 = self.circle_loss(outputs[3], sentence_sentiment_labels)
                    loss5 = self.circle_loss(outputs[4], sentence_emoji_labels)
                else:
                    loss4 = criterion2(outputs[3].view(-1),
                                       sentence_sentiment_labels.view(-1))  # /self.opt.sentiment_class
                    loss5 = criterion2(outputs[4].view(-1), sentence_emoji_labels.view(-1))  # /self.opt.emoji_class
                loss6 = criterion(outputs[5], rating)
                # loss = loss1 + loss2 + loss3
                if outputs[0] is not None:
                    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                else:
                    loss = loss4 + loss5

                if self.opt.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if self.opt.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()

                n_correct[0] += (torch.argmax(outputs[0], -1) == word_sentiment_labels).sum().item()
                n_correct[1] += (torch.argmax(outputs[1], -1) == word_sentiment_word_labels).sum().item()
                n_correct[2] += (torch.argmax(outputs[2], -1) == word_emoji_labels).sum().item()
                has_pos_sample_sentiment = torch.sum(sentence_sentiment_labels, dim=-1) != 0
                n_correct[3] += (outputs[3].ge(0.5).int()[has_pos_sample_sentiment] == sentence_sentiment_labels.int()[
                    has_pos_sample_sentiment]).sum().item()
                has_pos_sample_emoji = torch.sum(sentence_emoji_labels, dim=-1) != 0
                n_correct[4] += (outputs[4].ge(0.5).int()[has_pos_sample_emoji] == sentence_emoji_labels.int()[
                    has_pos_sample_emoji]).sum().item()
                n_correct[5] += (torch.argmax(outputs[5], -1) == rating).sum().item()

                n_total_word += masked_tokens.sum().item()
                n_total_sentence[0] += has_pos_sample_sentiment.sum().item()
                n_total_sentence[1] += has_pos_sample_emoji.sum().item()
                loss_total += loss.item() * len(text_bert_indices)
                n_total_loss += len(text_bert_indices)
                if global_step % self.opt.log_step == 0:
                    train_acc_word = n_correct[:3] / n_total_word
                    train_acc_sentiment_sentence = n_correct[3] / n_total_sentence[0]
                    if n_total_sentence[1] == 0:
                        train_acc_emoji_sentence = 0
                    else:
                        train_acc_emoji_sentence = n_correct[4] / n_total_sentence[1]
                    train_loss = loss_total / n_total_loss
                    train_acc_rating = n_correct[5] / n_total_loss
                    logger.info(
                        '{}/{}---loss: {:.4f}, acc: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(i_batch, batch_num, train_loss, train_acc_word[0], train_acc_word[1], train_acc_word[2], train_acc_sentiment_sentence, train_acc_emoji_sentence, train_acc_rating))

                if i_batch % int(batch_num/40) == 0 and i_batch != 0:
                    # print(i_batch, int(batch_num/20))
                    val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
                    logger.info('> val_acc: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f},val_f1: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(val_acc[0], val_acc[1], val_acc[2], val_acc[3], val_acc[4], val_acc[5], val_f1[0], val_f1[1], val_f1[2], val_f1[3], val_f1[4], val_f1[5]))
                    if val_f1[-1] > max_val_f1:
                        max_val_f1 = val_f1[-1]
                        if not os.path.exists('state_dict'):
                            os.mkdir('state_dict')
                        path = 'state_dict/{0}_{1}_val_f1_{2}_meanf1_{3}_Rating'.format(self.opt.model_name, self.opt.dataset,
                                                                     round(max_val_f1, 4), round(np.mean(val_f1), 4))
                        # torch.save(self.model.state_dict(), path)
                        torch.save(self.model.module.state_dict(), path)
                        path_bert = 'state_dict/bert_{0}_{1}_val_f1_{2}_meanf1_{3}_Rating'.format(self.opt.model_name, self.opt.dataset,
                                                                               round(max_val_f1, 4), round(np.mean(val_f1), 4))
                        if not os.path.exists(path_bert):
                            os.mkdir(path_bert)
                        self.model.module.bert.save_pretrained(path_bert)
                        logger.info('>> saved: {}'.format(path))
                    if np.mean(val_acc) > max_val_acc:
                        max_val_acc = np.mean(val_acc)
        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0
        n_f1, n_total_f1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0
        t_targets_all, t_outputs_all = None, [None, None, None, None, None, None]
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                if (test_flag and t_batch > 10) or t_batch > 1000:
                    break
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_word_sentiment_labels = t_sample_batched['word_sentiment_labels']#.to(self.opt.device)
                t_word_sentiment_word_labels = t_sample_batched['word_sentiment_word_labels']#.to(self.opt.device)
                t_word_emoji_labels = t_sample_batched['word_emoji_labels']#.to(self.opt.device)
                t_sentence_sentiment_labels = t_sample_batched['sentence_sentiment_labels']#.to(self.opt.device)
                t_sentence_emoji_labels = t_sample_batched['sentence_emoji_labels']#.to(self.opt.device)
                t_rating = t_sample_batched['rating']
                labels = [t_word_sentiment_labels, t_word_sentiment_word_labels, t_word_emoji_labels,
                          t_sentence_sentiment_labels, t_sentence_emoji_labels, t_rating]
                t_outputs = self.model(t_inputs)
                t_outputs = list(t_outputs)
                for i in range(len(t_outputs)):
                    t_outputs[i] = t_outputs[i].cpu()
                for i in range(3):
                    n_correct[i] += (torch.argmax(t_outputs[i], -1) == labels[i]).sum().item()
                for i in range(3, 5):
                    n_correct[i] += (t_outputs[i].ge(0.5).int() == labels[i].int()).sum().item()
                n_correct[5] += (torch.argmax(t_outputs[5], -1) == labels[5]).sum().item()
                n_total += len(t_outputs[0])

                if t_targets_all is None:
                    t_targets_all = labels
                    for i in range(6):
                        t_outputs_all[i] = t_outputs[i]
                else:
                    for i in range(6):
                        t_targets_all[i] = torch.cat((t_targets_all[i], labels[i]), dim=0)
                        t_outputs_all[i] = torch.cat((t_outputs_all[i], t_outputs[i]), dim=0)

                if t_batch % 50 == 0:
                    for i in range(6):
                        # t_targets_all[i] = t_targets_all[i]
                        # if t_targets_all[i].sum().item() == 0:
                        #     continue

                        if i < 3:
                            n_f1[i] += metrics.f1_score(t_targets_all[i].view(-1),
                                                   torch.argmax(t_outputs_all[i], -1).view(-1), average='macro')
                        if i == 3 or i == 4:
                            # print(t_targets_all[i].size(), t_outputs_all[i].ge(0.5).int().size())
                            n_f1[i] += metrics.f1_score(t_targets_all[i].view(-1), t_outputs_all[i].ge(0.5).int().view(-1),
                                                   average='macro')
                        if i == 5:
                            n_f1[i] += metrics.f1_score(t_targets_all[i], torch.argmax(t_outputs_all[i], -1), labels=[0, 1, 2, 3, 4], average='macro')
                    t_targets_all, t_outputs_all = None, [None, None, None, None, None, None]
                    n_total_f1 += 1

        acc = n_correct / n_total
        f1 = n_f1 / n_total_f1
        return acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        if self.opt.fp16:
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.opt.fp16_opt_level)
        if self.opt.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if self.opt.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.opt.local_rank], output_device=self.opt.local_rank,
                find_unused_parameters=True
            )

        self.opt.train_batch_size = self.opt.batch_size * max(1, self.opt.n_gpu)
        train_sampler = RandomSampler(self.trainset) if self.opt.local_rank == -1 else DistributedSampler(self.trainset)
        train_data_loader = DataLoader(dataset=self.trainset, sampler=train_sampler,
                                       batch_size=self.opt.train_batch_size, num_workers=8)
        test_sampler = SequentialSampler(self.testset)
        test_data_loader = DataLoader(dataset=self.testset, sampler=test_sampler, batch_size=self.opt.batch_size,
                                      num_workers=8)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, test_data_loader)
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
    parser.add_argument('--log_step', default=50, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
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