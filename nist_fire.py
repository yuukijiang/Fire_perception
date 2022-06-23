import read_data
import test_analize
from sklearn.metrics import confusion_matrix
from torchsummary import summary
import datetime
import time
import torch
from torch.autograd import Variable
# Variable是torch.autograd中很重要的类。它用来包装Tensor，将Tensor转换为Variable之后，可以装载梯度信息。
# Variable的参数为   “requires_grad”(True和False,True代表tensor变量需要计算梯度)
#                   “grad_fn(该变量是不是一个函数的输出值。若是，则grad_fn返回一个与该函数相关的对象，否则是None)”

import torch.optim as optim
# 一个optimizer优化器的库。

import torch.nn.functional as F
# nn.Xxx和nn.functional.xxx的实际功能是相同的， 运行效率也是近乎相同。nn.functional.xxx是函数接口，而nn.Xxx是nn.functional.xxx的类封装。
# 并且nn.Xxx都继承于一个共同祖先nn.Module。这导致nn.Xxx除了具有nn.functional.xxx功能之外，内部附带了nn.Module相关的属性和方法。
# nn.Xxx不需要自己定义和管理weight：而nn.functional.xxx需要你自己定义weight，每次调用的时候都需要手动传入weight, 不利于代码复用。

import sys

# sys模块提供了一系列有关Python运行环境的变量和函数。

sys.path.append("../../")
# 获得祖父级的文件目录

#from TCN.mnist_pixel.utils import data_generator
# 数据产生器，如果要迭代的值非常多，就要先将所有的值都放到列表中，而且即使迭代完了列表中所有的值，这些值也不会从内存中消失(至少还会存在一会)。
# 而且如果这些值只需要迭代一次就不再使用，那么这些值在内存中长期存在是没有必要的，所有就产生了产生器(Generator)的概念。
# 产生器只解决一个问题，就是让需要迭代的值不再常驻内存，也就是解决的内存资源消耗的问题。
# 为了解决这个问题，产生器也要付出一定的代价，这个代价就是产生器中的值只能访问一次，这也是产生器的特性。

#from TCN.mnist_pixel.model import TCN
import tcn
import numpy as np
import argparse

# argparse是python用于解析命令行参数和选项的标准模块，后面可以跟着不同的参数选项以实现不同的功能，argparse就可以解析命令行然后执行相应的操作。
# argparse：命令行选项、参数和子命令解析器

# path_train, path_test = 'E:\\fire_detection\\tcn\pth\\train', 'E:\\fire_detection\\tcn\pth\\test'
path_train, path_test = 'E:\\fire_detection\\tcn\pth\mk_process_data\\train', 'E:\\fire_detection\\tcn\pth\mk_process_data\\test'


parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
# ArgumentParser构造方法的调用都会使用"description="关键字参数,描述这个程度做什么以及怎么做。
# 在帮助消息中，描述会显示在命令行用法字符串和各种参数的帮助消息之间。
# 定义了一个解析器parser，第一步是创建一个ArgumentParser对象，ArgumentParser对象包含将命令行解析成Python数据类型所需的全部信息。
# 以下开始定义参数。

parser.add_argument('--batch_size', type=int, default=3, metavar='N',
                    help='batch size (default: 64)')
# 设定参数"--batch_size"              一次训练所选取的样本数目，大小需要调参。          小批量梯度下降法
# default - 不指定参数时的默认值。
# type - 命令行参数应该被转换成的类型。
# metavar - 在 usage 说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.
# help - 参数的帮助信息，当指定为 argparse.SUPPRESS 时表示不显示该参数的帮助信息.

parser.add_argument('--cuda', action='store_false',default=False,
                    help='use CUDA (default: False)')
# 设定参数"--cuda"              使用GPU加速，当使用了CUDA加速的时候，参数"--cuda"为false
# action - 命令行遇到参数时的动作，默认值是 store。
# store_true 是指触发action时为真，不触发则为假。

parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (default: 0.05)')
# 设定参数"dropout"防止训练的模型出现过拟合现象
# 一般都是0.5左右，但是试验了一下，发现在MNIST识别里面，dropout参数设置为0.05的准确率会比0.5的时候要高
# 当训练批次都取值epochs=100的时候
#       (1)dropout取0.5 的时候，准确率是97.09%
#       (2)dropout取0.05的时候，准确率是97.44%
# 模型在训练数据上损失函数较小，预测准确率较高；但是在测试数据上损失函数比较大，预测准确率较低。

parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
# 设定参数"clip_gradient"   (梯度裁剪) 的引入是为了处理gradient explosion(梯度爆炸)的问题。
# 当在一次迭代中权重的更新过于迅猛的话，很容易导致loss divergence。clip_gradient 的直观作用就是让权重的更新限制在一个合适的范围。
# 1.在solver中先设置一个clip_gradient
# 2.在前向传播与反向传播之后，我们会得到每个权重的梯度diff，这时不像通常那样直接使用这些梯度进行权重更新，
#       而是先求所有权重梯度的平方和sumsq_diff，如果sumsq_diff > clip_gradient，
#       则求缩放因子scale_factor = clip_gradient / sumsq_diff。这个scale_factor在(0,1)之间。
#       如果权重梯度的平方和sumsq_diff越大，那缩放因子将越小。
# 3.最后将所有的权重梯度乘以这个缩放因子，这时得到的梯度才是最后的梯度信息。

parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit (default: 20)')
# 设定参数"epoch"迭代次数            epochs被定义为向前和向后传播中所有批次的单次训练迭代。这意味着1个周期是整个输入数据的单次向前和向后传递。
# 随着训练迭代次数的增加，模型的准确率增长的速度逐渐变慢，直至逼近于一个常数
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 7)')
# 设定参数"ksize"卷积核尺寸
# 7×7大小
# 一般的TCN卷积核大小应该是2×2的，但是这里使用的7×7的卷积核，而且我也找不到这个卷积核的具体数值大小

parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 8)')
# 设定参数"levels"
# TCN的基本模块，包含8个部分，两个（卷积+修剪+relu+dropout）
# 即[25,25,25,25,25,25,25,25]

parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 10')
# 设定参数"log-interval"显示准确率
# 几个steps设置一次显示输出准确率，用于观察训练过程。

parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
# 设定参数"lr"初始的学习率的设定          把初始学习率设定为2*10^(-3)
# 1.一开始先设定一个很小的学习率，随着batch step变大，当损失函数不再下降，而是开始波动的时候，
# 拐点处的学习率设置为初始学习率(以梯度下降为例)
# 2.设定完初始学习率以后，先训练一段时间，到一定epoch后，损失开始不再下降而是波动，此时开始衰减学习率。
# 学习率(Learning rate)作为监督学习以及深度学习中重要的超参数，其决定着目标函数能否收敛到局部最小值以及何时收敛到最小值。
# 合适的学习率能够使目标函数在合适的时间内收敛到局部最小值。

parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
# 设定了优化器的类型--"AdamOptimizer"
# Adam 这个名字来源于自适应矩估计(Adaptive Moment Estimation)，
# 也是梯度下降算法的一种变形，但是每次迭代参数的学习率都有一定的范围，
# 不会因为梯度很大而导致学习率(步长)也变得很大，参数的值相对比较稳定。
# 首先，Adam中动量直接并入了梯度一阶矩(指数加权)的估计。
# 其次，相比于其他的梯度下降算法，
# Adam包括偏置修正，修正从原点初始化的一阶矩(动量项)和(非中心的)二阶矩估计。

parser.add_argument('--nhid', type=int, default = 24,
                    help='number of hidden units per layer (default: 25)')
# 设定参数"nhid"每层的隐藏单元数
# 这个参数和levels参数一起搭建起来了TCN模型的架构

parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1111)')
# 设定参数"seed"随机生成器种子         可以在调用其他随机模块函数之前调用此函数
# random.seed() 会改变随机生成器的种子；传入的数值用于指定随机数生成时所用算法开始时所选定的整数值，
# 如果使用相同的seed()值，则每次生成的随机数都相同；
# 如果不设置这个值，则系统会根据时间来自己选择这个值，此时每次生成的随机数会因时间的差异而有所不同。

parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')  ########################################################这个东西到底是干什么的？......
# 设定参数"permute"         维度的换位
# permute(A,[2,1,3])    若A是一个3维矩阵，此行代码交换矩阵A的第一维与第二维。
# store_true 是指触发action时为真，不触发则为假。

args = parser.parse_args()

###############################                 小结              ########################################
# 1.引入模块            import argparse
# 2，建立解析对象       parser = argparse.ArgumentParser()
# 3.增加属性            给***实例建立一个***属性
# 4.属性给与args实例：  把parser中设置的所有"add_argument"给返回到args子类实例当中，
#                       那么parser中增加的属性内容都会在args实例中，使用即可。
#                       args = parser.parse_args()


torch.manual_seed(args.seed)
# 为CPU设置种子用于生成随机数，以使得结果是确定的
# 手动设置种子一般可用于固定随机初始化的权重值，
# 这样就可以让每次重新从头训练网络时的权重的初始值虽然是随机生成的但却是固定的。
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# 如果发现可以使用CUDA加速但是还没有在模型里正确的打开GPU加速功能，就会显示这一行以提醒调整"--cuda"参数

# root = './data/mnist'
batch_size = args.batch_size
# default:64    一次训练所选取的样本数目
n_classes = 3
# 分类，0~9一共10个
input_channels = 3
# MNIST数据集的图片是28*28像素的，将其展开为一个一维的、长度为784的列向量
# seq_length = int(784 / input_channels)
seq_length = 20
# 序列长度，seq_length = int (784 / 1) = 784
epochs = args.epochs  # default:20
steps = 0
# 后面会用到在，这里只是定义一下



############################################                小结                  ######################################
# 把一些初始化的数值和解析器的一些属性都赋值出来



print(args)
# train_loader, test_loader = data_generator(root, batch_size)
# 数据迭代器，在utils.py里面
data_train, label_train = read_data.read_txt(path_train,window = 20,stride = 10,batch_size=batch_size)
data_test, label_test = read_data.read_txt(path_test,window = 20,stride = 10 ,batch_size=batch_size)
length_train = len(data_train)
length_test = len(data_test)
print(length_train)
print(length_test)
# permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
# 生成一个longTensor类型的长度为784的一维序列，使用了随机数种子使得每一次运行生成的这个序列与之前之后的序列都是一致的
# torch.Tensor默认是torch.FloatTensor是32位浮点类型数据，torch.LongTensor是64位整型
channel_sizes = [args.nhid] * args.levels
# 一个TCN基本块包含的通道数及层数 这里为[25,25,25,25,25,25,25,25]，即25*8
kernel_size = args.ksize
# 定义卷积核大小
model = tcn.TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)
# para = summary(model, (3, 20))
# print(para)
# print(model)
if args.cuda:
# 使用CUDA加速进行这个过程，读取和生成随机种子
    model.cuda()
    # permute = permute.cuda()
# 如果有GPU，把784*1的序列读入GPU
# 感觉"对象.cuda()"函数都是检测是不是可以放入GPU进行计算，来加速程序的运行，提高运行的效率

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
# getattr函数，用来返回对象optim的属性值，传递给优化器optimizer


def train(ep):
    all_mean_loss = []
    zip_train = zip(data_train, label_train)
    global steps
# Python中定义函数时，若想在函数内部对函数外的变量进行操作，就需要在函数内部声明其为global类型
    train_loss = 0
    epoch_loss = 0
# 训练误差,指的是在训练集上的误差
    model.train()
# 启用batch normalization和dropout

# batch normalization   网络训练过程中参数不断改变导致后续每一层输入的分布也发生变化，而学习的过程又要使每一层适应输入的分布，
#                       因此我们不得不降低学习率、小心地初始化。作者将分布发生变化称之为 internal covariate shift。
#                       论文《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》
#                       而BN就是通过一定的规范化手段，把每层神经网络任意神经元这个输入值的分布强行拉回到均值为0方差为1的标准正态分布，
#                       其实就是把越来越偏的分布强制拉回比较标准的分布，这样使得激活输入值落在非线性函数对输入比较敏感的区域，
#                       这样输入的小变化就会导致损失函数较大的变化，意思是这样让梯度变大，避免梯度消失问题产生，
#                       而且梯度变大意味着学习收敛速度快，能大大加快训练速度。
    for batch_idx, (data, target) in enumerate(zip_train):
        # enumerate()是python的内置函数，用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
        # 同时列出数据和数据下标，一般用在 for 循环当中。
        if args.cuda: data, target = data.cuda(), target.cuda()
        # 如果电脑支持GPU。那么就把data数据和target数据放进GPU里面
        data = data.view(-1, input_channels, seq_length)
        # if args.permute:
        #     data = data[:, :, permute]
        data, target = Variable(data), Variable(target)
        #   torch.autograd.Variable是Autograd的核心类，它封装了Tensor，并整合了反向传播的相关实现
        #   (tensor变成variable之后才能进行反向传播求梯度,用变量.backward()进行反向传播之后,var.grad中保存了var的梯度)
        #       data：存储了Tensor，是本体的数据
        #       grad：保存了data的梯度，本事是个Variable而非Tensor，与data形状一致
        #       grad_fn：指向Function对象，用于反向传播的梯度计算之用
        optimizer.zero_grad()  # 将梯度初始化为0
        output = model(data)

        loss = F.nll_loss(output, target.long(), reduction='sum')
        #   torch.nn.functional.nll_loss()函数
        #   torch.nn.functional.nll_loss(input, target, weight=None, size_average=True)
        #   常用于多分类任务，NLLLoss 函数输入input之前，需要对input进行log_softmax处理，即将input转换成概率分布的形式，并且取对数，底数为e
        #   - input - (N,C) C 是类别的个数
        #   - target - (N) 其大小是 0 <= targets[i] <= C-1
        #   - weight (Variable, optional) – 一个可手动指定每个类别的权重。如果给定的话，必须是大小为nclasses的Variable
        #   - size_average (bool, optional) – 默认情况下，是mini-batchloss的平均值，然而，如果size_average=False，则是mini-batchloss的总和。
        loss.backward()

        #   反向传播，计算当前梯度
        if args.clip > 0:
            #   clip的默认值为-1
            #   >0,发生了梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # 进行了梯度裁剪
        #   model.parameters():一个基于变量的迭代器，会进行归一化
        #   args.clip:梯度的最大范数
        optimizer.step()
        #   根据梯度更新网络参数
        train_loss += loss  # 计算损失值
        epoch_loss += loss
        steps += seq_length
        #   每个tensor张量里被seq_len=784，每784行组成一组，构成一个sequence，而每个tensor由batch_size决定有几个上面的sequence小组，
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            #   显示的功能，args.log_interval=10，当batch_idx为10的整数倍的时候print一下
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx, length_train,
                    100. * batch_idx / length_train, train_loss.item() / args.log_interval, steps))
            train_loss = 0

    mean_loss = float(epoch_loss / (length_train * batch_size))
    all_mean_loss.append(mean_loss)
    print('Train Epoch:{}\tmean_loss:{}'.format(ep, mean_loss))

#   每一个批次做完之后就要把train_loss的值给清零，以供下一epoch继续计算


def test():
    gai = []
    targets = []
    preds = []
    zip_test = zip(data_test, label_test)
    #   这个部分跟上面那个train()的结构有很大的相似的地方
    model.eval()
    # 不启用 BatchNormalization 和 Dropout，区别于train过程里用的是model.train()函数
    # 训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()，否则的话，有输入数据，即使不训练，它也会改变权值。
    # 这是model中含有batch normalization层所带来的的性质
    test_loss = 0
#   测试误差，指在测试集上产生的误差
    correct = 0
    with torch.no_grad():
        #   是一个上下文管理器，被该语句wrap起来的部分将不会track梯度。
        #   进入了eval阶段，即使不更新，但是在模型中所使用的dropout或者batch norm失效了，直接都会进行预测，
        #   而使用no_grad则设置让梯度Autograd设置为False(因为在训练中我们默认是True)，这样保证了反向过程为纯粹的测试，而不变参数
        for data, target in zip_test:
            if args.cuda:
        # GPU相关操作
                data, target = data.cuda(), target.cuda()
            # data = data.view(-1, input_channels, seq_length)
            # if args.permute:
            #     data = data[:, :, permute]  # 打乱data顺序
            data, target = Variable(data), Variable(target)
            #   volatile=True是Variable一个重要的标识，它能够将所有依赖它的节点全部设为volatile=True，
            #   使得volatile=True的节点不会求导，即使requires_grad=True(优先级高低地问题)，
            #   也不会进行反向传播，对于不需要反向传播的一些情况，volatile=True可以实现一定速度的提升，并节省显存，因为其不需要保存梯度
            output = model(data)

            test_loss += F.nll_loss(output, target.long(), reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            #   pred-预测值，从0~9
            #   1表示返回每行的最大值，即0~9内该图像最应该处于的概率最大的那个数字


            targets += target.numpy().tolist()
            preds += pred.numpy().tolist()
            gai += torch.exp(output).numpy().tolist()
            correct += pred.eq(target.data.view_as(pred).long()).cpu().sum()


        test_loss /= length_test * batch_size
        #   显示模块

        cm = confusion_matrix(np.array(targets).squeeze(), np.array(preds).squeeze())
        print(cm)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, length_test * batch_size,
            float(100 * correct) / (length_test * batch_size)))
        return np.array(targets).squeeze(),np.array(preds).squeeze(),np.array(gai).squeeze()


if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train_start = time.time()
        train(epoch)
        train_end = time.time()
        train_time_str = "train run time: %.4f seconds" % (train_end - train_start)
        print(train_time_str)
          # 在每一个epoch里
          # 先在训练集上面做完train的过程
        test_start = time.time()
        test_targets, test_preds , gailv= test()
        test_end = time.time()
        test_time_str = "test run time: %.4f seconds" % (test_end - test_start)
        print(test_time_str)
        if epoch == 30:
            with open('E:\\fire_detection\\tcn\pth\\tcn_result.txt', 'w') as f:
                for a, b in zip(test_targets, gailv):
                    f.write(str(a) + ' ' + str(b[0]) + ' ' + str(b[1]) + ' ' + str(b[2]) + "\r")
          # 然后去验证它的准确率
        if epoch % 10 == 0:
            lr /= 10
            #   随着学习的过程，调节学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

