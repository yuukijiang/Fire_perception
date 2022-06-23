import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import seaborn as sns
import itertools
import pandas as pd

def plot_confusion_matrix(cm,
                          cmap=plt.cm.OrRd, save_flg=False):
    plt.rc('font', family='Times New Roman')
    # classes = [str(i) for i in range(7)]
    classes = ['No-fire', 'Flaming', 'Smoldering']


    #plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title, fontsize=15, fontfamily='TimesNewRoman')
    plt.colorbar()
    plt.grid(False)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15, rotation=90, verticalalignment='center')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",verticalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    # if save_flg:
    #     plt.savefig("./confusion matrix of tcn-aap-svm.svg", dpi=500, format="svg", bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    sns.set()
    f, ax = plt.subplots()
    plt.rc('font', family='Times New Roman')
    mix_of_tcn_aap_svm = np.array([[388, 0, 2], [5, 139, 8], [5, 0, 251]])
    mix_of_tcn = np.array([[390, 0, 0], [7, 130, 15], [18, 0, 238]])
    mix_of_bp = np.array([[4036, 5, 1], [15, 1501, 0], [909, 0, 1648]])
    mix_of_lstm = np.array([[390, 0, 0], [8, 131, 12], [22, 0, 234]])

    plt.figure(figsize=(12, 9))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
    plt.subplot(2, 2, 1)
    plot_confusion_matrix(mix_of_tcn_aap_svm, save_flg=False)
    plt.title('(a)',y=-0.27, fontsize=13)
    plt.subplot(2, 2, 2)
    plot_confusion_matrix(mix_of_tcn, save_flg=False)
    plt.title('(b)', y=-0.27, fontsize=13)
    plt.subplot(2, 2, 3)
    plot_confusion_matrix(mix_of_bp, save_flg=False)
    plt.title('(c)', y=-0.27, fontsize=13)
    plt.subplot(2, 2, 4)
    plot_confusion_matrix(mix_of_lstm, save_flg=False)
    plt.title('(d)', y=-0.27, fontsize=13)
    plt.savefig("./confusion matrix.svg", dpi=500, format="svg", bbox_inches='tight')
    plt.show()



    # x = range(1, 31)
    # path = 'E:\\fire_detection\\tcn\pth\pic\\accandloss.txt'
    # accandloss = np.loadtxt(path, dtype=float, delimiter=' ')  # dtype是读出数据类型，delimiter是分隔符
    # newloss, newacc, oldloss, oldacc = np.split(accandloss, indices_or_sections=(1, 2, 3), axis=1)
    # print(newloss, newacc, oldloss, oldacc)
    # fig = plt.figure(figsize=(6, 3), dpi=80)
    # # plt.plot(x,oldloss, color='red', label='loss of TCN')
    # # plt.plot(x,newloss, color='blue',label = 'loss of improved TCN-GAP')
    # plt.plot(x,oldacc, color='red', label='acc of TCN')
    # plt.plot(x,newacc, color='blue',label = 'acc of improved TCN-GAP')
    # plt.legend()
    # plt.gca().spines['right'].set_color('none')
    # plt.gca().spines['top'].set_color('none')
    # plt.xlabel("epoch", fontsize=12)
    # plt.ylabel("test acc", fontsize=12)
    # plt.show()

    # x = range(1,21)
    # path1 = 'E:\\fire_detection\\tcn\pth\pic\sensordata_of_nofire.txt'
    # path2 = 'E:\\fire_detection\\tcn\pth\pic\sensordata_of_flaming.txt'
    # path3 = 'E:\\fire_detection\\tcn\pth\pic\sensordata_of_smoldering.txt'
    # sensordata1 = np.loadtxt(path1, dtype=float, delimiter=' ')  # dtype是读出数据类型，delimiter是分隔符
    # sensordata2 = np.loadtxt(path2, dtype=float, delimiter=' ')  # dtype是读出数据类型，delimiter是分隔符
    # sensordata3 = np.loadtxt(path3, dtype=float, delimiter=' ')  # dtype是读出数据类型，delimiter是分隔符
    # tem1, smo1, co1, ttem1, tsmo1, tco1 = np.split(sensordata1, indices_or_sections=(1, 2, 3, 4,5), axis=1)
    # tem2, smo2, co2, ttem2, tsmo2, tco2 = np.split(sensordata2, indices_or_sections=(1, 2, 3, 4, 5), axis=1)
    # tem3, smo3, co3, ttem3, tsmo3, tco3 = np.split(sensordata3, indices_or_sections=(1, 2, 3, 4, 5), axis=1)
    # # print(tem, smo, co, ttem, tsmo, tco)
    #
    # # 设置图片的右边框和上边框为不显示
    #
    # fig = plt.figure(figsize=(12, 9))
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.6)
    # plt.subplot(3, 2, 1)
    # plt.xlabel("point", fontsize=15)
    # plt.ylabel("Temperature (℃)", fontsize=15)
    # a1 = plt.scatter(x, tem1, color='red',marker='o')
    # b1 = plt.scatter(x, tem2, color='green',marker='x')
    # c1 = plt.scatter(x, tem3, color='blue',marker='*')
    # plt.gca().spines['right'].set_color('none')
    # plt.gca().spines['top'].set_color('none')
    # plt.xticks(range(1, 21))
    # plt.tick_params(labelsize=15)  # 刻度字体大小15
    # plt.title('(a)', y=-0.47, fontsize=15)
    #
    # plt.subplot(3, 2, 3)
    # plt.xlabel("point", fontsize=15)
    # plt.ylabel("Smoke (1/m)", fontsize=15)
    # a2 = plt.scatter(x, smo1, color='red', marker='o')
    # b2 = plt.scatter(x, smo2, color='green', marker='x')
    # c2 = plt.scatter(x, smo3, color='blue', marker='*')
    # plt.gca().spines['right'].set_color('none')
    # plt.gca().spines['top'].set_color('none')
    # plt.xticks(range(1, 21))
    # plt.tick_params(labelsize=15)  # 刻度字体大小15
    # plt.title('(c)', y=-0.47, fontsize=15)
    #
    # plt.subplot(3, 2, 5)
    # plt.xlabel("point", fontsize=15)
    # plt.ylabel("CO (Vol%)", fontsize=15)
    # a3 = plt.scatter(x, co1, color='red', marker='o')
    # b3 = plt.scatter(x, co2, color='green', marker='x')
    # c3 = plt.scatter(x, co3, color='blue', marker='*')
    # plt.gca().spines['right'].set_color('none')
    # plt.gca().spines['top'].set_color('none')
    # plt.xticks(range(1, 21))
    # plt.tick_params(labelsize=15)  # 刻度字体大小15
    # plt.title('(e)', y=-0.47, fontsize=15)
    #
    # plt.subplot(3, 2, 2)
    # plt.xlabel("point", fontsize=15)
    # plt.ylabel("Trend of \n Temperature", fontsize=15)
    # a11 = plt.scatter(x, ttem1, color='red', marker='o')
    # b11 = plt.scatter(x, ttem2, color='green', marker='x')
    # c11 = plt.scatter(x, ttem3, color='blue', marker='*')
    # plt.gca().spines['right'].set_color('none')
    # plt.gca().spines['top'].set_color('none')
    # plt.xticks(range(1, 21))
    # plt.tick_params(labelsize=15)  # 刻度字体大小15
    # plt.title('(b)', y=-0.47, fontsize=15)
    #
    # plt.subplot(3, 2, 4)
    # plt.xlabel("point", fontsize=15)
    # plt.ylabel("Trend of Smoke", fontsize=15)
    # a = plt.scatter(x, tsmo1, color='red', marker='o')
    # b = plt.scatter(x, tsmo2, color='green', marker='x')
    # c = plt.scatter(x, tsmo3, color='blue', marker='*')
    # plt.gca().spines['right'].set_color('none')
    # plt.gca().spines['top'].set_color('none')
    # plt.xticks(range(1, 21))
    # plt.tick_params(labelsize=15)  # 刻度字体大小15
    # plt.title('(d)', y=-0.47, fontsize=15)
    #
    # plt.subplot(3, 2, 6)
    # plt.xlabel("point", fontsize=15)
    # plt.ylabel("Trend of CO", fontsize=15)
    # a33 = plt.scatter(x, tco1, color='red', marker='o')
    # b33 = plt.scatter(x, tco2, color='green', marker='x')
    # c33 = plt.scatter(x, tco3, color='blue', marker='*')
    # plt.gca().spines['right'].set_color('none')
    # plt.gca().spines['top'].set_color('none')
    # plt.xticks(range(1,21))
    # plt.ylim(-1.1,1.1)
    # plt.tick_params(labelsize=15)  # 刻度字体大小15
    # plt.title('(f)', y=-0.47, fontsize=15)
    #
    # plt.legend((a1, b1, c1), ('Processed data of No-fire', 'Processed data of Flaming', 'Processed data of Smoldering'),
    #            loc = 'center', frameon=True,ncol=3, bbox_to_anchor=(-0.2, -0.6), fontsize=15)
    #
    #
    # plt.savefig("./precessed_data.svg", dpi=500, bbox_inches='tight',format="svg")
    # plt.show()