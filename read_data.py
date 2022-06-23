import glob
import torch
import numpy as np
import os
import random

def shuffle(a,k):
    b = 0
    c = []
    d = []
    l = []
    for i in range(0,len(a)):
        b = b + 1
        c.append(b)
    random.seed(1)
    random.shuffle(c)
    for j in c:
        d.append(a[j-1])
        l.append(k[j-1])
    return d,l

def read_txt(path,window,stride,batch_size):
    os.getcwd()
    os.chdir(path)
    files = glob.glob("*.txt")
    files.sort(key=lambda x: str(x[7:]))
    data = []
    label = []
    point_slide_data = []
    window_tile_data = []
    window_tile_label = []
    norm_data_line = []
    norm_data = []
    for filename in files:  #遍历文件夹里的每一个txt
        label_line = []
        data_line = []
        print(filename)
        f = open(filename)
        for i in f.readlines():
            j = i.split(' ')
            data_line.append([float(j[0]), float(j[1]), float(j[2])])   #取出三种传感器数据
            label_line.append(float(j[6]))  #取出标签数据
        label.append(label_line)
        data.append(data_line)
        f.close()
    for i in data:  #遍历每一个值，找最大最小值
        for j in i:
            point_slide_data.append(j)

    max_data = np.max(point_slide_data,axis = 0)
    min_data = np.min(point_slide_data,axis = 0)
    # max_data = np.array([1,1,1])
    # min_data = np.array([0,0,0])
    print(max_data)
    print(min_data)
    for i in data:  #归一化
        for j in i:
            norm_data_line.append((j - min_data) / (max_data - min_data))
        norm_data.append(norm_data_line)
        norm_data_line = []
    # norm_data = data
    for i in norm_data:
        print(len(i))
    for k in range(len(norm_data)):
        norm_data[k], label[k] = moving_sliding_window(norm_data[k], label[k], window, stride)
        norm_data[k] = torch.Tensor(np.float64(norm_data[k]))
        label[k] = torch.Tensor(np.float64(label[k]))
    for i in range(len(norm_data)):
        for j in norm_data[i]:
            grid = j
            grid = [[row[i] for row in grid] for i in range(len(grid[0]))]
            grid = torch.Tensor(grid)
            window_tile_data.append(grid)
        for k in label[i]:
            # k = torch.Tensor(k)
            window_tile_label.append(k)
    slide_data, slide_label = let_batch_size(window_tile_data, window_tile_label, batch_size, batch_size)
    # print(data)
    # print(label)
    # a = torch.stack(slide_data,0)
    # print(len(slide_label))
    return slide_data,slide_label

def moving_sliding_window(data,label,window,stride):
    data_out = []
    label_out = []
    data_out_line = []
    num = (len(data)-window) // stride + 1
    for i in range(num):
        for j in range(stride * i, window + stride * i):
            data_out_line.append(data[j])
        label_out.append(label[window + stride * i - 1])
        data_out.append(data_out_line)
        data_out_line = []
    return data_out,label_out

def let_batch_size(data1,label1,window,stride):
    data,label = shuffle(data1, label1)
    data_out = []
    label_out = []
    data_out_line = []
    label_out_line = []
    num = (len(data)-window) // stride + 1
    for i in range(num):
        for j in range(stride * i, window + stride * i):
            data_out_line.append(data[j])
            label_out_line.append(label[j])
        label_out_line = torch.stack(label_out_line, 0)
        data_out_line = torch.stack(data_out_line, 0)
        label_out.append(label_out_line)
        data_out.append(data_out_line)

        data_out_line = []
        label_out_line = []
    data_out = torch.stack(data_out, 0)
    label_out = torch.stack(label_out, 0)
    return data_out,label_out

if __name__ == '__main__':
    print(1)
    # path_train = 'train'
    # label0 = 0
    # label1 = 0
    # label2 = 0
    # data,label = read_txt(path_train,window = 20,stride = 10,batch_size=3)
    # a = zip(data,label)
    #
    # for i in label:
    #     for j in i:
    #         if j == 0:
    #             label0 = label0 + 1
    #         elif j == 1:
    #             label1 = label1 + 1
    #         elif j == 2:
    #             label2 = label2 + 1
    # print(label0,label1,label2)
    # print(data)
    #
    #
    # for batch_idx, (data,label) in enumerate(a):
    #     print(batch_idx)
    #     print(len(label))

