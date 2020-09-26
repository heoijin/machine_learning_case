from sklearn import linear_model
import numpy as np
from matplotlib import pyplot as plt


def train(xArr, yArr):
    '''
    训练模型
    :param xArr:
    :param yArr:
    :return:
    '''
    m, n = np.shape(xArr)
    xMat = np.mat(np.ones((m, n + 1)))
    x = np.mat(xArr)
    # 加第一列设为1，计算截距
    xMat[:, 1:n + 1] = x[:, 0:n]
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print('This matrix is sigular, cannot do inverse')
        return None
    ws = xTx.T * (xMat.T * yMat)
    return ws


def predict(xArr, ws):
    '''
    预测
    :param xArr:
    :param ws:
    :return:
    '''
    m, n = np.shape(xArr)
    # 加第一列设为1，计算截距
    xMat = np.mat(np.ones((m, n + 1)))
    x = np.mat(xArr)
    xMat[:, 1:n + 1] = x[:, 0:n]
    return xMat * ws


def main():
    '''
    使用方法：最小二乘法
    :return:
    '''
    x = [[1], [2], [3], [4]]
    y = [4.1, 5.9, 8.1, 10.1]
    # # <editor-fold desc="基础代码实现">
    # ws = train(x, y)
    # if isinstance(ws, np.ndarray):
    #     print(ws)
    #     print(predict([[5]], ws))
    #     plt.scatter(x, y, s=20)
    #     yHat = predict(x, ws)
    #     plt.plot(x, yHat, linewidth=2.0)
    #     plt.show()
    # # </editor-fold>

    # <editor-fold desc="调库实现方法">
    model = linear_model.LinearRegression()
    model.fit(x, y)
    print(model.intercept_, model.coef_)
    # intercept：2.0000000000000018
    # coef：[2.02]
    print(model.predict([[5]])) # 预测结果：[12.1]
    # </editor-fold>


if __name__ == '__main__':
    main()
