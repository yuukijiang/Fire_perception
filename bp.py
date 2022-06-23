import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class bp(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(bp, self).__init__()
        self.liner1 = nn.Linear(input_size, hidden_size)
        self.liner2 = nn.Linear(hidden_size, hidden_size)
        self.liner3 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.liner1(x)
        out = self.liner2(out)
        out = self.liner3(out)
        out = self.out(out)

        return F.log_softmax(out, dim=1)

def batch(data,batch_size):
    data_line = []
    data_block = []
    for i in range(1,len(data)):
        data_block.append(data[i])
        if(i % batch_size == 0):
            data_line.append(np.squeeze(data_block))
            data_block = []
    return torch.Tensor(data_line)

if __name__ == '__main__':
    input_size = 3
    hidden_size = 6
    output_size = 3
    model = bp(input_size, hidden_size,output_size)
    input = torch.rand(3,3)
    out= model(input)
    print(input)
    print(out)