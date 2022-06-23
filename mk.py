import numpy as np
import pymannkendall
import os
import glob

def mk_test(data):
    result = pymannkendall.original_test(data)
    return result.Tau

def single_col_mk_process(data, window = 16):
    stride = 1
    num1 = (len(data) - window) // stride + 1
    data_window = []
    data_window_block = []
    for i in range(num1):
        for j in range(stride * i, window + stride * i):
            data_window_block.append(data[j])
        data_window.append(data_window_block)
        data_window_block = []

    data_trend_pre = []
    for i in range(len(data)):
        if(i < window-1):
            data_trend_pre.append(0)
        else:
            data_trend_pre.append(data_window[i-window])
    data_trend = []
    for i in range(len(data_trend_pre)):
        if(i < window-1):
            data_trend.append(str(0))
        else:
            data_trend.append(str(mk_test(data_trend_pre[i])))
    return data_trend


if __name__ == '__main__':

    path = 'E:\\fire_detection\\tcn\pth\\train_pro_plus\\test_5.txt'
    datatest = np.loadtxt(path, dtype=float, delimiter=' ')
    # 3.划分数据与标签
    x1, x2, x3, z, y = np.split(datatest, indices_or_sections=(1, 2, 3, 6,), axis=1)
    # x为数据，y为标签，indices_or_sections表示从4列分割

    x1_trend = single_col_mk_process(x1)
    x2_trend = single_col_mk_process(x2)
    x3_trend = single_col_mk_process(x3)
    label = []
    dx1 = []
    for i in x1:
        dx1.append(str(float(i)))
    dx2 = []
    for i in x2:
        dx2.append(str(float(i)))
    dx3 = []
    for i in x3:
        dx3.append(str(float(i)))
    for i in y:
        label.append(str(int(i)))

    da = zip(dx1,dx2,dx3,x1_trend,x2_trend,x3_trend,label)

    with open('E:\\fire_detection\\tcn\pth\mk_process_data\\train\\train_19.txt', 'w') as f:
        for i in da:
            for j in i:
                f.write(j + ' ')
            f.write("\r")



    print('hello')
