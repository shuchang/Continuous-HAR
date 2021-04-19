import argparse

import torch
from torch import nn

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--output_size', type=int, default=6)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--drop_rate', type=float, default=0.5)
parser.add_argument('--bidirectional', type=bool, default=False)
parser.add_argument('--seq_len', type=int, default=256)
parser.add_argument('--overlap_factor', type=int, default=0.5)
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size

INPUT_SIZE = FLAGS.input_size
HIDDEN_SIZE = FLAGS.hidden_size
OUTPUT_SIZE = FLAGS.output_size
NUM_LAYERS = FLAGS.num_layers
DROP_RATE = FLAGS.drop_rate
BIDIRECTIONAL = FLAGS.bidirectional
NUM_DIRECTIONS = 2 if BIDIRECTIONAL else 1

SEQ_LEN = FLAGS.seq_len


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
        h_0 = torch.zeros(NUM_LAYERS*NUM_DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE).to(DEVICE)
        c_0 = torch.zeros(NUM_LAYERS*NUM_DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE).to(DEVICE)
        return h_0, c_0

    def forward(self, x):
        # self.hidden = self.init_hidden()
        # input: (batch, seq_len, input_size) output: (batch, seq_len, num_directions*hidden size)
        x, (h_n, c_n) = self.lstm(x, self.init_hidden())
        x = x.contiguous().view(-1, HIDDEN_SIZE*NUM_DIRECTIONS)
        x = self.act(self.dropout(self.fc(x)))
        x = self.dropout(self.fc2(x))
        outputs = x.view(BATCH_SIZE, SEQ_LEN, OUTPUT_SIZE)[:,-1,:]
        return outputs
