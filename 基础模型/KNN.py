# <editor-fold desc="KNN基础用库">
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
# </editor-fold>

# <editor-fold desc="非调用库实现KNN">
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import numpy as np
import operator


# </editor-fold>


def basic_KNN(x_train, x_test, y_train, y_test):
    '''
    基础KNN模型（调用库）
    TODO 涉及步骤：
    1.
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    '''
    clf = neighbors.KNeighborsClassifier(5)  # 设置临近值为5
    clf.fit(x_train, y_train)  # 训练模型
    print(clf.score(x_test, y_test))  # 给模型评分：0.8947368421052632

    print(f'预测结果：{clf.predict([x_test[0]])},实际结果： {y_test[0]}')  # .predict方法需要放入2维数据作为参数，因此当预测自变量为一维数据时，需要外接一个空列表
    # 预测结果：[0],实际结果： 0

    proba = clf.predict_proba([x_test[0]])  # 计算因变量属于各分类的概率
    print(f'因变量属于第0类的概率{proba[0][0]}\n因变量属于第1类的概率{proba[0][1]}')
    # 返回一个二维数组，每一个元素为每一个因变量在各个分类上的概率
    # 因变量属于第0类的概率0.6
    # 因变量属于第1类的概率0.4


def classify(inx, dataSet, labels, k):
    '''
    目标：对实例进行预测
    步骤：
        1. 使用欧氏距离/马氏距离计算距离
        2. 选取邻近样例中最多的类别作为该实例的预测
    :param inx: 测试集自变量
    :param dataSet: 训练集自变量
    :param labels: 训练集因变量
    :param k: 邻近个数
    :return:
    '''
    # <editor-fold desc="马氏距离实现(未优化)">
    S = np.cov(dataSet.T)  # 协方差矩阵，用于计算马氏距离
    SI = np.linalg.inv(S)  # 协方差矩阵的逆矩阵
    distances = np.array(distance.cdist(dataSet, [inx], 'mahalanobis', VI=SI)).reshape(-1)
    # </editor-fold>
    # 计算欧氏距离，求训练集的测试集各自的自变量之间的距离
    distances = np.array(distance.cdist(dataSet, [inx], 'euclidean').reshape(-1))  # reshape(-1)为将任意维度的数组转化为一维数组
    sortedDistIndicies = distances.argsort()  # 将元素从小到大排序，并获取其原索引，用于labels排序
    classCount = {}  # 计算出k个点中，各类别出现的次数
    for i in range(k):  # 访问距离最近的5个实例
        voteILabel = labels[sortedDistIndicies[i]]  # 提取对应的训练集因变量
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1  # 获取字典内对应元素，如果没有元素返回0

    sortedClassCount = sorted(
        classCount.items(),  # 以列表嵌套元组的形式返回，[(键，值),(键，值)]
        key=operator.itemgetter(1),  # 用于获取对象的第1个域的值,此处作用为将值作为大小排序的依据，没有此行则以键的大小作为排序依据
        reverse=True  # 倒序
    )  # 选出k个点中，出现次数最多的分类

    return sortedClassCount[0][0]  # 返回最多的分类


def main():
    data = datasets.load_breast_cancer()
    X = data.data  # 自变量
    Y = data.target  # 因变量
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

    # 调库实现方法
    basic_KNN(x_train, x_test, y_train, y_test)

    # 非调库实现方法
    ret = [classify(x_test[i], x_train, y_train, 5) for i in range(len(x_test))]
    print(accuracy_score(y_test, ret))
    # 模型得分
    # 欧氏距离：0.8947368421052632，马氏距离：0.8596491228070176


if __name__ == '__main__':
    main()
