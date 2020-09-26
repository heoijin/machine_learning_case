from sklearn.datasets import load_iris # 鸢尾花数据集
from sklearn.model_selection import train_test_split # 数据切分工具
from sklearn import tree # 决策树
import pydotplus # 作图工具
import io

def main():
    iris=load_iris()
    X=iris.data
    y=iris.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    clf=tree.DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train,y_train)
    print(f"score:{clf.score(X_test,y_test)}")

    # 生成决策树
    dot_data=io.StringIO()
    tree.export_graphviz(
        clf,out_file=dot_data,
        feature_names=iris.feature_names,
        filled=True,rounded=True,
        impurity=False
    )
    graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
    open('a.jpg','wb').write(graph.create_jpg())

if __name__ == '__main__':
    main()