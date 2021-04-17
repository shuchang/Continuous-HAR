import numpy as np
from scipy.io import loadmat
import os
from provider import shuffle_data, sliding_window


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/discrete/discrete_data.mat')
LABEL_DIR = os.path.join(BASE_DIR, 'data/discrete/discrete_label.mat')

TRAIN_DIR = os.path.join(BASE_DIR, 'data/train.npy')
TEST_DIR = os.path.join(BASE_DIR, 'data/test.npy')


OVERLAP_FACTOR = 0.5
SEQ_LEN = 256


def make_data(data_path, label_path):
    data = loadmat(data_path)['Doppler_data']
    label = loadmat(label_path)['label']
    data_size, feat_dims, seq_len = data.shape
    raw_x = np.zeros([feat_dims, seq_len, data_size], dtype=np.float32)
    raw_y = np.zeros([1, data_size], dtype=np.int32)
    for i in range(data_size):
        raw_x[:, :, i] = data[i, :, :]
        raw_y[:, i] = label[i][0] - 1  # convert the indexing format
    data_x = raw_x.transpose((2, 1, 0))  # (data_size, seq_len, feat_dims)
    data_y = raw_y.transpose((1, 0))


    # train test split
    shuffled_x, shuffled_y, _ = shuffle_data(data_x, data_y)
    data_size = shuffled_x.shape[0]
    train_data_ratio = 0.8    # TODO: 0.9
    train_data_len = round(train_data_ratio*data_size)
    train_x = shuffled_x[:train_data_len, :, :]
    train_y = shuffled_y[:train_data_len, :]
    test_x = shuffled_x[train_data_len:, :, :]
    test_y = shuffled_y[train_data_len:, :]

    # data segmentation
    # overlap_factor = OVERLAP_FACTOR
    # window_len = SEQ_LEN
    # train_x, train_y = sliding_window(train_x, train_y, overlap_factor, window_len) # seq_len down, data_size up, feat_dim -
    # test_x, test_y = sliding_window(test_x, test_y, overlap_factor, window_len) # seq_len down, data_size up, feat_dim -

    train_dict = {'data': train_x, 'label': train_y}
    test_dict = {'data': test_x, 'label': test_y}

    np.save(TRAIN_DIR, train_dict)
    np.save(TEST_DIR, test_dict)
    print("Data saved ...")

if __name__ == "__main__":
    make_data(DATA_DIR, LABEL_DIR)