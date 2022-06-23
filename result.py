from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')
import matplotlib as mpl

def rocandauc(y_true_hot1, y_pred_hot1, y_true_hot2, y_pred_hot2, y_true_hot3, y_pred_hot3, y_true_hot4, y_pred_hot4,):

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr1, tpr1, thresholds1 = roc_curve(y_true_hot1.squeeze().ravel(), y_pred_hot1.squeeze().ravel())
    roc_auc1 = auc(fpr1, tpr1)
    plt.rc('font', family='Times New Roman')
    # FPR就是横坐标,TPR就是纵坐标
    plt.plot(fpr1, tpr1, c='r', lw=2, alpha=0.7, label=u'TCN-AAP-SVM (AUC = %.2f%%)' % (roc_auc1 * 100))
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=15)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=15)
    plt.xlabel('FPR', fontsize=15)
    plt.ylabel('TPR', fontsize=15)
    # plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=15)
    # plt.title(u'鸢尾花数据Logistic分类后的ROC和AUC', fontsize=17)
    # plt.show()

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr2, tpr2, thresholds2 = roc_curve(y_true_hot2.squeeze().ravel(), y_pred_hot2.squeeze().ravel())
    roc_auc2 = auc(fpr2, tpr2)

    plt.rc('font', family='Times New Roman')
    # FPR就是横坐标,TPR就是纵坐标
    plt.plot(fpr2, tpr2, c='b', lw=2, alpha=0.7, label=u'TCN (AUC = %.2f%%)' % (roc_auc2 * 100))
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=15)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=15)
    plt.xlabel('FPR', fontsize=15)
    plt.ylabel('TPR', fontsize=15)
    # plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=15)
    # plt.title(u'鸢尾花数据Logistic分类后的ROC和AUC', fontsize=17)
    # plt.show()

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr3, tpr3, thresholds3 = roc_curve(y_true_hot3.squeeze().ravel(), y_pred_hot3.squeeze().ravel())
    roc_auc3 = auc(fpr3, tpr3)

    plt.rc('font', family='Times New Roman')
    # FPR就是横坐标,TPR就是纵坐标
    plt.plot(fpr3, tpr3, c='y', lw=2, alpha=0.7, label=u'BP neural network (AUC = %.2f%%)' % (roc_auc3 * 100))
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=15)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=15)
    plt.xlabel('FPR', fontsize=15)
    plt.ylabel('TPR', fontsize=15)
    # plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=15)
    # plt.title(u'鸢尾花数据Logistic分类后的ROC和AUC', fontsize=17)
    # plt.show()

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr4, tpr4, thresholds4 = roc_curve(y_true_hot4.squeeze().ravel(), y_pred_hot4.squeeze().ravel())
    roc_auc4 = auc(fpr4, tpr4)

    plt.rc('font', family='Times New Roman')
    # FPR就是横坐标,TPR就是纵坐标
    plt.plot(fpr4, tpr4, c='g', lw=2, alpha=0.7, label=u'LSTM (AUC = %.2f%%)' % (roc_auc4 * 100))
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=15)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=15)
    plt.xlabel('FPR', fontsize=15)
    plt.ylabel('TPR', fontsize=15)
    # plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=15)
    # plt.title(u'鸢尾花数据Logistic分类后的ROC和AUC', fontsize=17)
    plt.savefig("./rocandauc.svg", dpi=500, bbox_inches='tight',format="svg")
    plt.show()



path = './result/tcn_app_svm_result.txt'
tcn_app_svm_result = np.loadtxt(path, dtype=float, delimiter=' ')   # dtype是读出数据类型，delimiter是分隔符
y_true_of_tcn_app_svm, y_pred_of_tcn_app_svm = np.split(tcn_app_svm_result, indices_or_sections=(1,), axis=1)

path = './result/tcn_result.txt'
tcn_app_svm_result = np.loadtxt(path, dtype=float, delimiter=' ')   # dtype是读出数据类型，delimiter是分隔符
y_true_of_tcn, y_pred_of_tcn = np.split(tcn_app_svm_result, indices_or_sections=(1,), axis=1)

path = './result/bp_result.txt'
tcn_app_svm_result = np.loadtxt(path, dtype=float, delimiter=' ')   # dtype是读出数据类型，delimiter是分隔符
y_true_of_bp, y_pred_of_bp = np.split(tcn_app_svm_result, indices_or_sections=(1,), axis=1)

path = './result/lstm_result.txt'
tcn_app_svm_result = np.loadtxt(path, dtype=float, delimiter=' ')   # dtype是读出数据类型，delimiter是分隔符
y_true_of_lstm, y_pred_of_lstm = np.split(tcn_app_svm_result, indices_or_sections=(1,), axis=1)

y_true1 = y_true_of_tcn_app_svm
y_pred1 = y_pred_of_tcn_app_svm

y_true_hot1 = label_binarize(y_true1, np.arange(3))  #装换成类似二进制的编码
# y_pred_hot1 = label_binarize(y_pred1, np.arange(3))  #装换成类似二进制的编码
y_pred_hot1 = y_pred1

y_true2 = y_true_of_tcn
y_pred2 = y_pred_of_tcn

y_true_hot2 = label_binarize(y_true2, np.arange(3))  #装换成类似二进制的编码
# y_pred_hot2 = label_binarize(y_pred2, np.arange(3))  #装换成类似二进制的编码
y_pred_hot2 = y_pred2

y_true3 = y_true_of_bp
y_pred3 = y_pred_of_bp

y_true_hot3 = label_binarize(y_true3, np.arange(3))  #装换成类似二进制的编码
# y_pred_hot3 = label_binarize(y_pred3, np.arange(3))  #装换成类似二进制的编码
y_pred_hot3 = y_pred3

y_true4 = y_true_of_lstm
y_pred4 = y_pred_of_lstm

y_true_hot4 = label_binarize(y_true4, np.arange(3))  #装换成类似二进制的编码
# y_pred_hot4 = label_binarize(y_pred4, np.arange(3))  #装换成类似二进制的编码
y_pred_hot4 = y_pred4

rocandauc(y_true_hot1, y_pred_hot1, y_true_hot2, y_pred_hot2, y_true_hot3, y_pred_hot3, y_true_hot4, y_pred_hot4)

target_names = ['no fire', 'flaming', 'smoldering']
print(classification_report(y_true1, y_pred1, target_names=target_names))