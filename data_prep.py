import numpy as np
from scipy.io import loadmat
import os
from provider import shuffle_data, sliding_window


def make_discrete_data(data_path, train_dir, test_dir):
    """
    """
    data = loadmat(data_path)['Doppler']
    label = loadmat(data_path)['Label']

    data_x, data_y = [[] for _ in range(2)]
    data_size = len(data)
    for i in range(data_size):
        raw_x = data[i][0].T
        seq_len, feat_dims = raw_x.shape
        dop_x = np.abs(raw_x) # extract doppler
        raw_y = np.zeros([seq_len, 1], dtype=np.int32)
        raw_y[:, :] = label[i][0] - 1 # convert the indexing format
        slide_x, slide_y = sliding_window(dop_x, raw_y, overlap=0.9, win_len=50)
        data_x.append(slide_x)
        data_y.append(slide_y)
    data_x = np.vstack(data_x)
    data_y = np.vstack(data_y) # (data_size, seq_len, feat_dims)

    # train test split
    shuffled_x, shuffled_y, _ = shuffle_data(data_x, data_y)
    data_size = shuffled_x.shape[0]
    train_data_ratio = 0.8    # TODO: 0.9
    train_data_len = round(train_data_ratio*data_size)
    train_x = shuffled_x[:train_data_len, :, :]
    train_y = shuffled_y[:train_data_len, :]
    test_x = shuffled_x[train_data_len:, :, :]
    test_y = shuffled_y[train_data_len:, :]

    train_dict = {'data': train_x, 'label': train_y}
    test_dict = {'data': test_x, 'label': test_y}
    np.save(train_dir, train_dict)
    np.save(test_dir, test_dict)
    print(f"Data saved to {train_dir}, {test_dir}")


if __name__ == "__main__":
    make_discrete_data('datasets/processed/discrete2.mat',
                        'data/Doppler_train.npy', 'data/Doppler_test.npy')