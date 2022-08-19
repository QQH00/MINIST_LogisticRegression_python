#导入数值计算的基础库
import numpy as np
import struct
import time
import warnings
import matplotlib.pyplot as plt
## 导入画图库
import matplotlib.pyplot as plt
import seaborn as sns
## 导入逻辑回归模型函数
from sklearn.linear_model import LogisticRegression

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


def logisticRegression(n):
    # 调用逻辑回归模型
    lr_clf = LogisticRegression(max_iter=n)
    warnings.filterwarnings("ignore")
    # 用逻辑回归模型拟合构造的数据集
    lr_clf = lr_clf.fit(train_x, train_y)  # 其拟合方程为 y=w0+w1*x1+w2*x2
    predict_y = lr_clf.predict(test_x)
    m = test_x.shape[0]
    errorCount = 0
    for i in range(m):
        if test_y[i] != predict_y[i]:
            errorCount += 1

    error_rate = float(errorCount) / m
    return error_rate


if __name__ == '__main__':
    time1 = time.time()
    train_x, test_x, train_y, test_y = loadDataSet()
    time2 = time.time()
    print("read data cost", time2 - time1, "second")

    error = []
    for n in range(1, 100, 1):
        error.append(logisticRegression(n))

    x = range(1, 100, 1)
    plt.xlabel('Number of iterations')
    plt.ylabel('error rate')
    plt.plot(x, error, color='k')
    plt.show()
    plt.save('./iteration.png')
