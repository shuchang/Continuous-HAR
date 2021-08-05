#!/usr/bin/env python

import os

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from provider import shuffle_data, split_data, sliding_window


def main():
    data_dir = 'datasets/preprocessed/'
    filenames = os.listdir(data_dir)
    for filename in tqdm(filenames):
        if filename == 'Continuous1_48.mat':
            data_x, data_y = prep_data(data_dir + filename, False)
            data_saver(data_x, data_y, 'data/train_' + filename[:-3] + 'npy', 'data/test_' + filename[:-3] + 'npy')
        else:
            data_x, data_y = prep_data(data_dir + filename, True)
            data_saver(data_x, data_y, 'data/train_' + filename[:-3] + 'npy', 'data/test_' + filename[:-3] + 'npy')
    print('All data have been saved!\n')


def prep_data(data_path, is_discrete):
    """ Prepare data for training and testing using sliding window method\n
        Parameters:\n
            `data_path`: path for preprocessed data;\n
            `is_discrete`: if the input data is discrete\n
        Return:\n
            `data_x`: data after sliding window;\n
            `data_y`: labels after sliding window
    """
    data = loadmat(data_path)['Doppler']
    label = loadmat(data_path)['Label']

    data_x, data_y = [[] for _ in range(2)]
    data_size = len(data)
    for i in range(data_size):
        # extract doppler data
        raw_x = data[i][0].T
        dop_x = np.abs(raw_x)
        # extract labels
        seq_len = raw_x.shape[0]
        raw_y = np.zeros([seq_len, 1], dtype=np.int32)
        # convert the indexing format of labels
        if is_discrete == True:
            raw_y[:, :] = label[i][0] - 1
        else:
            raw_y[:, :] = label[i][0].T - 1

        slide_x, slide_y = sliding_window(dop_x, raw_y, overlap=0.9, win_len=50)
        data_x.append(slide_x)
        data_y.append(slide_y)
    data_x = np.vstack(data_x)
    data_y = np.vstack(data_y) # (data_size, seq_len, feat_dims)
    return data_x, data_y


def data_saver(data_x, data_y, train_dir, test_dir):
    """ Do the shuffle and split and save data and labels into dictionaries
    """
    shuffled_x, shuffled_y, _ = shuffle_data(data_x, data_y)
    train_x, train_y, test_x, test_y = split_data(shuffled_x, shuffled_y, split_ratio=0.8)
    # save as dictionary
    train_dict = {'data': train_x, 'label': train_y}
    test_dict = {'data': test_x, 'label': test_y}
    np.save(train_dir, train_dict)
    np.save(test_dir, test_dict)

    print("\ntrain data shape:", train_x.shape)
    print("\ntest data shape:", test_x.shape)
    print(f"\nData saved to {train_dir}, {test_dir}\n")


if __name__ == "__main__":
    # main()
    data_x, data_y = prep_data('datasets/preprocessed/Continuous1_48.mat', False)
    data_saver(data_x, data_y, 'data/train_Continuous1_48.npy', 'data/test_Continuous1_48.npy')
