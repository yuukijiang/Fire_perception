import torch.nn as nn
from torch.nn.utils import weight_norm
import torch
import torch.nn.functional as F

class Chomp1d(nn.Module): #剪枝,一维卷积后会出现多余的padding。
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        tensor.contiguous()会返回有连续内存的相同张量
        有些tensor并不是占用一整块内存，而是由不同的数据块组成
        tensor的view()操作依赖于内存是整块的，这时只需要执行
        contiguous()函数，就是把tensor变成在内存中连续分布的形式
        本函数主要是增加padding方式对卷积后的张量做切边而实现因果卷积
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module): #时序模块,两层一维卷积，两层Weight_Norm,两层Chomd1d，非线性激活函数为Relu,dropout为0.2
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        #定义第一个膨胀卷积层，膨胀是指dilation
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        #根据第一个卷积层的输出与padding大小实现因果卷积
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)
        #在先前的输出结果上添加激活函数与dropout完成第一个卷积
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)

        #将卷积模块的所有组件通过sequential方法依次堆叠在一起，具体来说的话网络结构是一层一层的叠加起来的，nn库里有一个类型
        #叫做sequence序列，它是一个容器类，可以在里面添加一些基本的模块。
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU()
        self.init_weights()
        #正如先前提到的卷积前与卷积后的通道数不一定相同，所以如果通道数不一样，那么需要对输入x做一个逐元素的一维卷积
        #以使得他的维度与前面两个卷积相等
    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        #初始化方法是从均值为0，标准差为0.01的正态分布采样
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)#残差模块
        return self.relu(out + res)

#时序卷积模块,使用for循环对8层隐含层，每层25个节点进行构建。模型如下。
#其中*layer不是c中的指针，困惑了笔者一段时间，之后查看资料知道 * 表示迭代器拆分layers为一层层网络

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)   # *作用是将输入迭代器拆成一个个元素

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)

#TCN模块,创新点1D·FCN，最后采用softmax进行分类

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        o = self.tcn(inputs)  # input should have dimension (N, C, L)
        y2 = torch.nn.functional.adaptive_avg_pool2d(o, (3, 1))
        return F.log_softmax(y2[:, :, -1], dim=1), y2[:, :, -1]
        # a = self.linear(o[:, :, -1])  # 增加一个维度,1D·FCN
        # return F.log_softmax(a, dim=1), a
