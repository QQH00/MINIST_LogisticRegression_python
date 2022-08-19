from numpy import *
import numpy as np
from numpy import random
import time
from scipy.special import expit
import matplotlib.pyplot as plt
import struct


# 读取图片
def read_image(file_name):
    # 先用二进制方式把文件都读进来
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中
    offset = 0
    head = struct.unpack_from('>IIII', file_content, offset)  # 取前4个整数，返回一个元组
    offset += struct.calcsize('>IIII')
    imgNum = head[1]  # 图片数
    rows = head[2]  # 宽度
    cols = head[3]  # 高度

    images = np.empty((imgNum, 784))  # empty，是它所常见的数组内的所有元素均为空，没有实际意义，它是创建数组最快的方法
    image_size = rows * cols  # 单个图片的大小
    fmt = '>' + str(image_size) + 'B'  # 单个图片的format

    for i in range(imgNum):
        images[i] = np.array(struct.unpack_from(fmt, file_content, offset))
        offset += struct.calcsize(fmt)
    return images


# 读取标签
def read_label(file_name):
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中

    head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
    offset = struct.calcsize('>II')

    labelNum = head[1]  # label数
    bitsString = '>' + str(labelNum) + 'B'  # fmt格式：'>47040000B'
    label = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    return np.array(label)


def loadDataSet():
    train_x_filename = "train-images-idx3-ubyte"
    train_y_filename = "train-labels-idx1-ubyte"
    test_x_filename = "t10k-images-idx3-ubyte"
    test_y_filename = "t10k-labels-idx1-ubyte"
    train_x = read_image(train_x_filename)
    train_y = read_label(train_y_filename)
    test_x = read_image(test_x_filename)
    test_y = read_label(test_y_filename)

    return train_x, test_x, train_y, test_y


# train_model(train_x, train_y, theta, learning_rate, iteration,numClass)
def train_model(train_x, train_y, theta, learning_rate, numClass):  # theta是n+1行的列向量
    m = train_x.shape[0]
    train_x = np.insert(train_x, 0, values=1, axis=1)

    for k in range(numClass):
        real_y = np.zeros((m, 1))
        index = train_y == k  # index中存放的是train_y中等于0的索引
        real_y[index] = 1  # 在real_y中修改相应的index对应的值为1，先分类0和非0

        temp_theta = theta[:, k].reshape((785, 1))
        # 求概率
        h_theta = expit(np.dot(train_x, temp_theta)).reshape((60000, 1))
        # 似然函数（取对数版）为了梯度下降而不是上升而取负
        # J_theta[j, k] = (np.dot(np.log(h_theta).T, real_y) + np.dot((1 - real_y).T, np.log(1 - h_theta))) / (-m)
        # 梯度下降
        temp_theta = temp_theta + learning_rate * np.dot(train_x.T, (real_y - h_theta))

        theta[:, k] = temp_theta.reshape((785,))

    error.append(predict(test_x, test_y, theta))

    return theta  # 返回的theta是n*numClass矩阵


def predict(test_x, test_y, theta):  # 这里的theta是学习得来的最好的theta，是n*numClass的矩阵
    errorCount = 0
    test_x = np.insert(test_x, 0, values=1, axis=1)
    m = test_x.shape[0]

    h_theta = np.dot(test_x, theta)
    h_theta_max_postion = h_theta.argmax(axis=1)  # 获得每行的最大值的label
    for i in range(m):
        if test_y[i] != h_theta_max_postion[i]:
            errorCount += 1

    error_rate = float(errorCount) / m
    return error_rate


def show(error, iteration):
    x = range(0, iteration, 1)
    plt.xlabel('Number of iterations')
    plt.ylabel('error rate')
    plt.plot(x, error, color='k')
    plt.show()
    plt.savefig('./iteration.png')


if __name__ == '__main__':
    print("Start reading data...")
    time1 = time.time()
    train_x, test_x, train_y, test_y = loadDataSet()
    time2 = time.time()
    print("read data cost", time2 - time1, "second")

    numClass = 1000
    iteration = 10000
    learning_rate = 0.000000003
    n = test_x.shape[1] + 1
    error = []

    theta = random.random(size=(n, numClass))  # theta=np.random.rand(n,1)#随机构造n*numClass的矩阵,因为有numClass个分类器，所以应该返回的是numClass个列向量（n*1）

    print("Start training data...")
    for i in range(iteration):
        theta_new = train_model(train_x, train_y, theta, learning_rate, numClass)
        if i % 10 == 0:
            print(predict(test_x, test_y, theta))
    time3 = time.time()
    print("train data cost", time3 - time2, "second")

    print("Start predicting data...")
    final_error_rate = predict(test_x, test_y, theta_new)
    time4 = time.time()
    print("predict data cost", time4 - time3, "second")
    print(final_error_rate)

    show(error, iteration)


