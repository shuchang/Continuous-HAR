# -*- coding:UTF-8 -*-
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from model import LSTM
from provider import MyDataset, cal_acc, loadDataFile


torch.manual_seed(42) 

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'data/train.npy')
TEST_DIR = os.path.join(BASE_DIR, 'data/test.npy')
LOG_DIR = os.path.join(BASE_DIR,'log')
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
MODEL_DIR = os.path.join(LOG_DIR,'model_params_Doppler_LSTM.pt')
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_evaluate.txt'), 'w')


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--output_size', type=int, default=6)
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
OUTPUT_SIZE = FLAGS.output_size


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def load_data(train_filepath, test_filepath):
    train_x, train_y = loadDataFile(train_filepath)
    test_x, test_y = loadDataFile(test_filepath)

    train_loader = torch.utils.data.DataLoader(dataset=MyDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=MyDataset(test_x, test_y), batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    return train_loader, test_loader


def eval_one_epoch(test_loader, lstm, loss_func):

    lstm.eval()

    epoch_loss = 0
    epoch_acc = 0
    num_batches = len(test_loader)

    current_label = torch.zeros([num_batches, BATCH_SIZE])
    pred_label = torch.zeros(current_label.shape)
    confusion_matrix = torch.zeros([OUTPUT_SIZE, OUTPUT_SIZE])

    with torch.no_grad():
        for step, (test_x_tensor, test_y_tensor) in enumerate(test_loader):
            test_x_tensor = test_x_tensor.to(DEVICE)
            test_y_tensor = test_y_tensor.to(DEVICE)

            test_y_tensor = test_y_tensor.view(-1)
            test_outs = lstm(test_x_tensor).view(-1, OUTPUT_SIZE)
            loss = loss_func(test_outs, test_y_tensor)
            acc = cal_acc(test_outs, test_y_tensor)

            epoch_loss += loss.item()
            epoch_acc += acc

            current_label[step, :] = test_y_tensor
            pred_label[step, :] = torch.max(test_outs, 1)[1]

        for t, p in zip(current_label.view(-1), pred_label.view(-1)):
            confusion_matrix[t.long(),  p.long()] += 1

        print(confusion_matrix)
        print(confusion_matrix.diag()/confusion_matrix.sum(1)) # per class accuracy
        print("macro F1 score:", f1_score(current_label.view(-1), pred_label.view(-1), average='macro'))
        print("macro F1 score:", f1_score(current_label.view(-1), pred_label.view(-1), average='micro'))

    log_string('test loss: %f' % (epoch_loss / num_batches))
    log_string('test accuracy: %f' % (epoch_acc / num_batches))


def evaluate():

    _, test_loader = load_data(TRAIN_DIR, TEST_DIR)

    lstm = LSTM().to(DEVICE)
    checkpoint = torch.load(MODEL_DIR)
    lstm.load_state_dict(checkpoint)

    loss_func = nn.CrossEntropyLoss().to(DEVICE)

    eval_one_epoch(test_loader, lstm, loss_func)


if __name__ == "__main__":
    evaluate()
    LOG_FOUT.close()
