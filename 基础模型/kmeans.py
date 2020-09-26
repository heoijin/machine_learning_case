from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X,y=make_blobs(n_samples=100,random_state=150)
# 创建训练集，样本量为100，样本特征数默认为2，随机生成器的种子数位150
# X为[[ 9.34936052,  6.77089003],[ 7.86655194, -4.1701109 ],...,]
y_pred=KMeans(n_clusters=3).fit_predict(X)
# 训练数据，分3个簇
plt.scatter(X[:,0],X[:,1],c=y_pred)
# X[:,0] X轴
# X[:,1] Y轴
plt.show()