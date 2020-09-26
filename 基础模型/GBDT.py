'''
使用Sklearn库中的GDBT方法实现波士顿房价预测功能
    - 使用5成决策树（每个基模型最多5成）
    - 200次迭代
'''
from sklearn import ensemble
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 读取Sklearn库中自带的数据集
boston = datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target,
    test_size=0.2, random_state=13
)

params = {
    'n_estimators': 200,
    'max_depth': 5,
    'min_samples_split': 5,
    'learning_rate': 0.01,
    'loss': 'ls',
    'random_state': 0
}

clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)
print('MSE:{0:0.2f}'.format(mean_squared_error(y_test, clf.predict(X_test))))

test_score = []
for i, y_pred in enumerate(clf.staged_predict(X_test)):
    # 计算测试集误差
    test_score.append(clf.loss_(y_test, y_pred))
plt.figure(figsize=(16, 9), dpi=160)
plt.plot(clf.train_score_, 'y-')
plt.plot(test_score, 'b-')
plt.show()
