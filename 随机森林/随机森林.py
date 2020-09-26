'''
目标：通过变量[年龄，使用时长，支付情况，流量，通话情况]预测宽带客户是否会续费
数据：broadband.csv
    -核心特征：BROADBAND（是否续费）
模型：bagging -> 随机森林
步骤：
    1.使用装袋法在行列上进行随机抽样
    2.组合多个决策树分类器
    3.随机方法构成单个决策树分类器
        - 学习集：从原训练集中通过有放回抽样得到的自助样本
        - 随机选出参与构建改决策树的变量，参与变量数通常大大小于可用变量数
    4.单个决策树在产生学习集合和确定参与变量后，使用CART算法计算，不剪枝
    5.分类结果取决于多个决策树分类器简单多数选举
调参：
    - 相关阅读：https://www.cnblogs.com/juanjiang/p/11003369.html
    -
'''
import pandas as pd
import sklearn.tree as tree
# 模型评估
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
# 使用交叉网络搜索来优化决策树模型，边训练边优化
from sklearn.model_selection import GridSearchCV
import sklearn.ensemble as ensemble  # 集成学习


def read_csv():
    '''
    步骤：
    1. 读取CSV数据
    2，拆分特征和变量
    :return:
    '''
    df = pd.read_csv('broadband.csv')
    # BROADBAND:{0：908，1：206}
    y = df['BROADBAND']
    # columns第0个为客户ID，不属于本次模型的有用特征，故丢弃
    x = df.iloc[:, 1:-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=12345)
    return (x_train, x_test, y_train, y_test)


def basic_tree():
    '''
    构建基础决策树
    :return:
    '''
    param_grid = {
        'criterion': ['entropy', 'gini'],  # 不纯度计算方法['信息熵','基尼系数']
        'max_depth': [2, 3, 4, 5, 6, 7, 8],  # 限制数的最大深度，超过设定深度的树枝全部剪掉
        'min_samples_split': [4, 8, 12, 16, 20, 24, 28]  # 一个节点必须要包含多少个训练样本，这样节点才允许分枝
    }
    clf = tree.DecisionTreeClassifier()  # 实例化一棵树
    # 传入模型、网络搜索参数、评估指标、cv交叉验证次数，此处仅定义
    clfcv = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=4  # 交叉验证次数
    )
    return clfcv


def train_predit(_func, x_train, x_test, y_train, y_test):
    '''
    训练模型
    :param _func:
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    '''
    _func.fit(X=x_train, y=y_train)
    # 进行预测
    test_est = _func.predict(x_test)
    print('决策树精准度：')
    print(metrics.classification_report(y_test, test_est))
    fpr_test, tpr_test, _ = metrics.roc_curve(y_test, test_est)
    print('决策树 AUC：{0:.4f}'.format(metrics.auc(fpr_test, tpr_test)))


def precision_adjustment(param_grid, x_test, x_train, y_test, y_train):
    '''
    步骤：
    1. 生成随机森林
    2. 调用train_predit（）训练随机森林并打印结果
    3. 打印最优参数
    :param param_grid:
    :param x_test:
    :param x_train:
    :param y_test:
    :param y_train:
    :return:
    '''
    rfc = ensemble.RandomForestClassifier()
    rfc_cv = GridSearchCV(
        estimator=rfc, param_grid=param_grid,
        scoring='roc_auc', cv=4
    )
    train_predit(rfc_cv, x_train, x_test, y_train, y_test)
    print(f'最佳参数为：{rfc_cv.best_params_}')


def main():
    x_train, x_test, y_train, y_test = read_csv()
    clfcv = basic_tree()  # 基础决策树模型
    train_predit(clfcv, x_train, x_test, y_train, y_test)
    # 决策树 AUC：0.6022。AUC 大于0.5是最基本的要求，因此此模型的精度比较差

    # 第一次尝试生成随机森林
    param_grid = {
        'criterion': ['entropy', 'gini'],
        'max_depth': [5, 6, 7, 8],
        'n_estimators': [11, 13, 15],  # 决策树个数
        'max_features': [0.3, 0.4, 0.5],  # 每棵决策树使用的变量占比
        'min_samples_split': [4, 8, 12, 16]  # 叶子的最小拆分样本量
    }
    precision_adjustment(param_grid, x_test, x_train, y_test, y_train)
    # 最佳参数为：{'criterion': 'gini', 'max_depth': 8, 'max_features': 0.5, 'min_samples_split': 4, 'n_estimators': 13}

    # 第二次生成随机森林
    param_grid = {
        'criterion': ['entropy', 'gini'],
        'max_depth': [7, 8, 10, 12, 14, 16],  # 最优参数在左边界，应该调大最大值范围
        'n_estimators': [11, 13, 15, 17, 19],  # 决策树个数，最优参数在右边界，应该向右调整最大值范围
        'max_features': [0.4, 0.5, 0.6, 0.7],  # 每棵决策树使用的变量占比，最优参数在右边界，应该向右调整最大值范围
        'min_samples_split': [2, 3, 4, 8, 12, 16]  # 叶子的最小拆分样本量，最优参数在左边界，应该向左调整取值范围
    }
    precision_adjustment(param_grid, x_test, x_train, y_test, y_train)


if __name__ == '__main__':
    main()