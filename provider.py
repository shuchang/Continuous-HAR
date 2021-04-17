import numpy as np
import torch
from torch.utils.data import Dataset

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