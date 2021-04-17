# -*- coding:UTF-8 -*-
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn

from model import LSTM
from provider import loadDataFile, MyDataset, cal_acc

torch.manual_seed(42) 

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'data/train.npy')
TEST_DIR = os.path.join(BASE_DIR, 'data/test.npy')

LOG_DIR = 'log'
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--max_epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--output_size', type=int, default=6)
FLAGS = parser.parse_args()

LR = FLAGS.learning_rate
MAX_EPOCH = FLAGS.max_epoch
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


def train_one_epoch(epoch, train_writer, train_loader, lstm, loss_func, optimizer):
    lstm.train()

    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0

    for step, (train_x_tensor, train_y_tensor) in enumerate(train_loader):  # max step is num_batch
        train_x_tensor = train_x_tensor.to(DEVICE)
        train_y_tensor = train_y_tensor.to(DEVICE)

        train_y_tensor = train_y_tensor.view(-1)
        train_outs = lstm(train_x_tensor).view(-1, OUTPUT_SIZE)        
        loss = loss_func(train_outs, train_y_tensor)
        acc = cal_acc(train_outs, train_y_tensor)

        epoch_loss += loss.item()
        epoch_acc += acc
        num_batches += 1

        optimizer.zero_grad()  # clear gradient for each batch
        loss.backward()
        optimizer.step()

    train_writer.add_scalar('Loss/train', epoch_loss / num_batches, epoch)
    train_writer.add_scalar('Accuracy/train', epoch_acc / num_batches, epoch)

    log_string('train loss: %f' % (epoch_loss / num_batches))
    log_string('train accuracy: %f' % (epoch_acc / num_batches))


def eval_one_epoch(epoch, test_writer, test_loader, lstm, loss_func):
    lstm.eval()

    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0

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
            num_batches += 1

    test_writer.add_scalar('Loss/test', epoch_loss / num_batches, epoch)
    test_writer.add_scalar('Accuracy/test', epoch_acc / num_batches, epoch)

    log_string('test loss: %f' % (epoch_loss / num_batches))
    log_string('test accuracy: %f' % (epoch_acc / num_batches))


def train():
    train_writer = SummaryWriter(os.path.join(LOG_DIR, 'train21-64-uni'))
    test_writer = SummaryWriter(os.path.join(LOG_DIR, 'test21-64-uni'))

    train_loader, test_loader = load_data(TRAIN_DIR, TEST_DIR)

    lstm = LSTM().to(DEVICE)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss().to(DEVICE)

    for epoch in range(MAX_EPOCH):
        log_string('**** EPOCH %3d ****' % (epoch))
        sys.stdout.flush()

        train_one_epoch(epoch, train_writer, train_loader,
                        lstm, loss_func, optimizer)
        eval_one_epoch(epoch, test_writer, test_loader, lstm, loss_func)

    # save model parameters to files
    torch.save(lstm.state_dict(), 'model_params.pt')


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
