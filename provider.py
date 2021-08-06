import numpy as np
import torch
from torch.utils.data import Dataset


def sliding_window(data_x, data_y, overlap, win_len):
    """ Use the sliding window method to segment data into small fragments\n
        Parameters:\n
            `data_x`: input data with shape (seq_len, feat_dims);\n
            `data_y`: input label with shape (seq_len, 1);\n
            `overlap`: the proportion of overlap in window between two neighbor windows;\n
            `win_len`: the time length of output data and label\n
        Return:\n
            `slide_x`: segmented data with shape (num_win, win_len, feat_dims);\n
            `slide_y`: segmented label with shape (num_win, win_len, 1)
    """
    seq_len, feat_dims = data_x.shape
    label_dim = data_y.shape[1]

    for_len = round(win_len*(1 - overlap)) # forward length
    num_win = int(1 + (seq_len - win_len)/for_len)
    slide_x = np.zeros([num_win, win_len, feat_dims], dtype=np.float16)
    slide_y = np.zeros([num_win, win_len, label_dim], dtype=np.int16)

    for i in range(num_win):
        slide_x[i, :, :] = data_x[for_len*i : win_len + for_len*i, :]
        slide_y[i, :, :] = data_y[for_len*i : win_len + for_len*i, :]
    return slide_x, slide_y


def shuffle_data(data, labels):
    """ Shuffle data and labels\n
        Parameters:\n
            data and labels\n
        Return:\n
            shuffled data, labels and shuffled indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx, ...], idx


def split_data(data_x, data_y, split_ratio):
    """ Split data into training set and testing set
    """
    data_size = data_x.shape[0]
    train_data_len = round(split_ratio*data_size)
    train_x = data_x[:train_data_len, :, :]
    train_y = data_y[:train_data_len, :]
    test_x = data_x[train_data_len:, :, :]
    test_y = data_y[train_data_len:, :]
    return train_x, train_y, test_x, test_y


def load_npy(npy_filepath):
    f = np.load(npy_filepath, allow_pickle=True)
    data = f.item().get('data')
    label = f.item().get('label')
    return (data, label)


def loadDataFile(filepath):
    return load_npy(filepath)


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


def cal_acc(outs, y):
    """ Calculate the accuracy of one batch on CPU\n
        Parameters:\n
            `outs`: model outputs with shape (batch_size*seq_len, output_size) in tensor;\n
            `y`: ground truth labels with shape (batch_size*seq_len, 1) in tensor\n
        Return:\n
            `acc`: classification accuracy in float
    """
    pred = torch.max(outs, 1)[1].data.cpu().numpy()
    y = y.data.cpu().numpy()
    acc = float((pred == y).astype(int).sum()) / float(y.size)
    return acc
