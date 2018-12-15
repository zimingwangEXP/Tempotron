from Tempotron import Tempotron
from keras.datasets import mnist
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt
import cv2


def Encoding2():
    return;


# 只对灰度图进行编码
def Encoding1(t_max, thred, img):
    bck = [];
    alpha = (np.exp(t_max) - 1) / thred;
    for x in img:
        for y in x:
            bck.append([t_max - np.log(alpha * y + 1)])
    return bck


def LoadMinist():
    global train_size, test_size
    (X_train_multi, y_train_multi), (X_test_multi, y_test_multi) = mnist.load_data();
    X_train_bin = []
    y_train_bin = []
    X_test_bin = []
    y_test_bin = []
    #将minist数据集中的1,7手写体单独拿出来，构造成一个二分类的数据集
    for i in range(len(y_train_multi)):
        if (y_train_multi[i] == 1 or y_train_multi[i] == 7):
            X_train_bin.append(X_train_multi[i])
            y_train_bin.append(y_train_multi[i]==1)
    for i in range(len(y_test_multi)):
        if (y_test_multi[i] == 1 or y_test_multi[i] == 7):
            X_test_bin.append(X_test_multi[i])
            y_test_bin.append(y_test_multi[i]==1)
    #将二分类数据集规模减小
    X_train_small = X_train_bin[:train_size]
    plt.imshow(X_train_small[4])
    plt.show()
    y_train_small = y_train_bin[:train_size]
    X_test_small = X_test_bin[:test_size]
    y_test_small = y_test_bin[:test_size]
    train_spike = []
    test_spike = []
    for i in range(train_size):
        train_spike.append((Encoding1(500, 255, X_train_small[i]), y_train_small[i]))
    for i in range(test_size):
        test_spike.append((Encoding1(500, 255, X_test_small[i]), y_test_small[i]))
    return  (train_spike,test_spike)

if __name__ == '__main__':
    global train_size, test_size
    train_size = 10
    test_size = 2
    np.random.seed(0);
    efficacies = 1.8 * np.random.random(28 * 28) - 0.50
    (train_spike, test_spike)=LoadMinist()
    work=Tempotron(efficacies,0,10,2.5,1,20);
    work.Train(train_spike)
    # work.PlotVT(0,500,train_spike[4][0])
    #     # work.Train([train_spike[4]]);
    #     # work.PlotVT(0,500,train_spike[4][0]);
    #     # print(train_spike[4][1])
    #     # plt.show()
    cnt=0;
    for i in range(test_size):
      if((work.ComputeMembranePotential(test_spike[i][0],work.ComputeTmax1(test_spike[i][0]))>1)==test_spike[i][1]):
          cnt+=1;
    print('accucy at test data is %f%%' %(cnt/test_size));

