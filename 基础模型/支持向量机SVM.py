from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt


def main():
    '''
    目标：多分类预测
    数据集：鸢尾花
    核函数：
        - 高斯核
        - 线性核
    :return:
    '''
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # clf = svm.SVC(C=0.8, kernel='rbf', gamma=1)  # 高斯核，松弛度C(大写)为0.8
    clf=svm.SVC(C=0.5,kernel='linear') # 线性核，松弛度为0.5
    clf.fit(X_train, y_train.ravel())
    print(f'train pred:{round(clf.score(X_train, y_train), 3)}')  # 对训练集打分
    print(f'test prtd:{round(clf.score(X_test, y_test), 3)}')  # 对测试集打分
    print(clf.support_vectors_)  # 支持向量列表，从中看切分边界
    print('-' * 20)
    print(clf.n_support_)  # 每个类别支持向量的个数
    plt.plot(X_train[:, 0], X_train[:, 1], 'o', color='#bbbbbb')
    plt.plot(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], 'o')
    plt.show()


if __name__ == '__main__':
    main()


