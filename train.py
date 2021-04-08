# -*- coding:UTF-8 -*-
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.io import loadmat
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import Dataset

torch.manual_seed(42) 

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/discrete/discrete_data.mat')
LABEL_DIR = os.path.join(BASE_DIR, 'data/discrete/discrete_label.mat')

LOG_DIR = 'log'
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_epoch', type=int, default=500)
parser.add_argument('--input_size', type=int, default=256)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--output_size', type=int, default=6)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--drop_rate', type=float, default=0.5)
parser.add_argument('--bidirectional', type=bool, default=True)
FLAGS = parser.parse_args()


LR = FLAGS.learning_rate
BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch

SEQ_LEN = 256
OVERLAP_FACTOR = 0.5

INPUT_SIZE = FLAGS.input_size
HIDDEN_SIZE = FLAGS.hidden_size
OUTPUT_SIZE = FLAGS.output_size
NUM_LAYERS = FLAGS.num_layers
DROP_RATE = FLAGS.drop_rate
BIDIRECTIONAL = FLAGS.bidirectional
NUM_DIRECTIONS = 2 if BIDIRECTIONAL else 1


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def load_data(data_path, label_path):
    data = loadmat(data_path)['Doppler_data']
    label = loadmat(label_path)['label']
    data_size, feat_dims, seq_len = data.shape
    raw_x = np.zeros([feat_dims, seq_len, data_size], dtype=np.float32)
    raw_y = np.zeros([1, seq_len, data_size], dtype=np.int32)
    for i in range(data_size):
        raw_x[:, :, i] = data[i, :, :]
        raw_y[:, :, i] = label[i][0] - 1  # convert the indexing format
    data_x = raw_x.transpose((2, 1, 0))  # (data_size, seq_len, feat_dims)
    data_y = raw_y.transpose((2, 1, 0))

    # overlap_factor = OVERLAP_FACTOR
    # window_len = SEQ_LEN
    # data_x, data_y = sliding_window(data_x, data_y, overlap_factor, window_len)

    # data_size = data_x.shape[0]
    # train_data_ratio = 0.9
    # train_data_len = round(train_data_ratio*data_size)
    # train_x = data_x[:train_data_len, :, :]
    # train_y = data_y[:train_data_len, :, :]
    # test_x = data_x[train_data_len:, :, :]
    # test_y = data_y[train_data_len:, :, :]

    # train test split
    shuffled_x, shuffled_y, _ = shuffle_data(data_x, data_y)
    data_size = shuffled_x.shape[0]
    train_data_ratio = 0.8    # TODO: 0.9
    train_data_len = round(train_data_ratio*data_size)
    train_x = shuffled_x[:train_data_len, :, :]
    train_y = shuffled_y[:train_data_len, :, :]
    test_x = shuffled_x[train_data_len:, :, :]
    test_y = shuffled_y[train_data_len:, :, :]

    # data segmentation
    overlap_factor = OVERLAP_FACTOR
    window_len = SEQ_LEN
    # train_x, train_y = sliding_window(train_x, train_y, overlap_factor, window_len) # seq_len down, data_size up, feat_dim -
    # test_x, test_y = sliding_window(test_x, test_y, overlap_factor, window_len) # seq_len down, data_size up, feat_dim -


    # # train data augmentation
    # train_x_ud = np.flip(train_x, axis=0)
    # train_x = np.concatenate((train_x, train_x_ud), axis=0)
    # train_y = np.concatenate((train_y, train_y), axis=0)


    train_loader = torch.utils.data.DataLoader(dataset=MyDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=MyDataset(test_x, test_y), batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    return train_loader, test_loader


# def train_transforms(data):
#     """ Augment the train set using torchvision transforms
#     Parameters:
#         data: batch image with (B, C, H, W) shape
#     Return:
#     """
#     data = np.expand_dims(data, axis=1)
#     transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(p=1),
#     ])
#     data_aug = transform(data)
#     return data_aug


def sliding_window(data_x, data_y, overlap_factor, window_len):
    """ Use the sliding window method to segment the original data into small fragments
        Parameters:
            data_x: input data with shape (data_size, seq_len, feat_dims)
            data_y: input label with shape (data_size, seq_len, 1)
            overlap_factor: the proportion of overlap in window between two neighbor windows
            window_len: the time length of output data and label
        Return:
            new_x: segmented data with shape (new_size, win_len, feat_dims)
            new_y: segmented label with shape (new_size, win_len, 1)
    """
    data_size, seq_len, feat_dims = data_x.shape
    label_dim = data_y.shape[2]

    forward_len = round(window_len*(1 - overlap_factor))
    num_window = int(1 + (seq_len - window_len)/forward_len)
    new_size = data_size*num_window
    new_x = np.zeros([new_size, window_len, feat_dims], dtype=np.float32)
    new_y = np.zeros([new_size, window_len, label_dim], dtype=np.int32)

    idx = 0
    for i in range(data_size):
        for j in range(num_window):
            new_x[idx, :, :] = data_x[i, forward_len*
                                      j: window_len + forward_len*j, :]
            new_y[idx, :, :] = data_y[i, forward_len*
                                      j: window_len + forward_len*j, :]
            idx += 1
    return new_x, new_y


def shuffle_data(data, labels):
    """ Shuffle data and labels
        Input:
            data:
            labels:
        Return:
            shuffled data, labels and shuffled indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx, ...], idx


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.FloatTensor(self.data[index])
        label = torch.LongTensor(self.label[index]) # nn.CrossEntropyLoss requires targets with Tensor Long type
        return data, label

    def __len__(self):
        return len(self.data)


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROP_RATE,
            bidirectional=BIDIRECTIONAL
        )

        # fully connected layer (num_directions*hidden_size)
        self.fc = nn.Linear(HIDDEN_SIZE*NUM_DIRECTIONS, HIDDEN_SIZE) # double hidden size for bidirectional
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.dropout = nn.Dropout(p=DROP_RATE)
        self.act = nn.ReLU()

    # initialize hidden layer
    # ## TODO: study the differences between randn and zero initialization
    def init_hidden(self):
        h_0 = torch.randn(NUM_LAYERS*NUM_DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE).to(DEVICE)
        c_0 = torch.randn(NUM_LAYERS*NUM_DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE).to(DEVICE)
        return h_0, c_0

    def forward(self, x):
        # self.hidden = self.init_hidden()
        # input: (batch, seq_len, input_size) output: (batch, seq_len, num_directions*hidden size)
        x, (h_n, c_n) = self.lstm(x, self.init_hidden())
        x = x.contiguous().view(-1, HIDDEN_SIZE*NUM_DIRECTIONS)
        x = self.act(self.dropout(self.fc(x)))
        x = self.dropout(self.fc2(x))
        outputs = x.view(BATCH_SIZE, SEQ_LEN, OUTPUT_SIZE)
        return outputs


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


def cal_acc(outs, y):
    """ Calculate the accuracy of one batch on CPU
        Parameters:
            outs: model outputs with shape (batch_size*seq_len, output_size) in tensor
            y: ground truth labels with shape (batch_size*seq_len, 1) in tensor
        Return:
            acc: classification accuracy in float
    """
    pred = torch.max(outs, 1)[1].data.cpu().numpy()
    y = y.data.cpu().numpy()
    acc = float((pred == y).astype(int).sum()) / float(y.size)
    return acc


def train():
    train_writer = SummaryWriter(os.path.join(LOG_DIR, 'train18'))
    test_writer = SummaryWriter(os.path.join(LOG_DIR, 'test18'))

    train_loader, test_loader = load_data(DATA_DIR, LABEL_DIR)

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
    # torch.save(lstm.state_dict(), 'model_params.pt')


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
