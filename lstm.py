import torch
import torch.nn as nn
import torch.nn.functional as F
# 实现一个num_layers层的LSTM-RNN

class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(lstm, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                  batch_first=True, dropout = 0.5)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        out = self.out(r_out[:, -1, :])

        return F.log_softmax(out, dim=1)


if __name__ == '__main__':
    input_size = 6
    hidden_size = 24
    num_layers = 8
    batch_size = 3
    output_size = 3
    model = lstm(input_size, hidden_size, num_layers, output_size)
    # input (seq_len, batch, input_size) 包含特征的输入序列，如果设置了batch_first，则batch为第一维
    input = torch.rand(3,20, 6)
    out= model(input)
    print(input)
    print(out)



