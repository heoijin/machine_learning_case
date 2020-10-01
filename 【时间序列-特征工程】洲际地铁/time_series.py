"""
项目名称：时间序列预测法
项目重点：
    1. 时间序列的特征工程 [ 月份 ， 本月第几日 ， 是否工作日 ， 时刻（早、晚高份/中午/凌晨） ]
    2. 基础GBM（梯度提升算法）预测
    3. 对比GBDT、XGBoost、LightGBM的性能
    4. TODO:使用交叉网络搜索来优化决策树模型，边训练边优化
相关阅读：
    - 原文：https://blog.csdn.net/deephub/article/details/108415495
"""
import time
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics, ensemble
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import seaborn as sns


def read_data():
    '''
    :shape:()
    :columns holiday object:美国法定假日、区域假日、博览会等
    :columns temp:平均温度（开尔文）
    :columns rain_1h:每小时降雨量（毫米）
    :columns snow_1h:每小时降雪（毫米）
    :columns clouds_all:云层情况（百分比）
    :columns weather_main:当前天气的分类描述（简要）
    :columns weather_description:当前天气的分类描述（详细）
    :columns date_time:时间序列数据
    :columns traffic_volume:每小时I-94 ATR 301记录的西行交通量（本文预测目标）
    :return: 
    '''
    df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')
    print(df.info())
    print(df.head())
    df['date_time'] = pd.to_datetime(df['date_time'])
    return df


def characteristics_the_engineer(df):
    '''
    特征工程：
        - 不变：
            1. temp
            2. rain_1h
            3. snow_1h
            4. clouds_all
        - 新增：
            1. months：月份
            2. day_of_month：日
            3. hours：时间
            4. weekday:是否工作日
        - 仅OneHot：
            1. is_holiday:是否节假日
            2. weathers：简要天气描述
    :param df:
    :return:
    '''
    months = df['date_time'].dt.month
    day_of_month = df['date_time'].dt.day
    hours = df['date_time'].dt.hour

    # 获取日期所在星期数，并进行OneHot编码
    week = pd.get_dummies(df['date_time'].dt.day_name())

    # 制作时间段标签：分箱 + 统一标签（因为分箱的标签不能重复） + OneHot Encode
    hour_label = ['midnight_1', 'dawn', 'morning','morring_rush', 'noon', 'afternoon','evening_peak', 'evening', 'midnight']
    daypart = pd.get_dummies(
        pd.cut(hours, bins=[0, 1, 5, 7, 9, 13, 17, 19, 21, 24], labels=hour_label)
            .apply(lambda x: 'midnight' if x == 'midnight' else x)
    )[hour_label[1:]]

    # 制作工作日标签
    is_weekday = df['date_time'].dt.dayofweek.apply(lambda x: 1 if x < 6 else 0)
    is_holiday = df['holiday'].apply(lambda x: 0 if x == 'None' else 1)

    weathers = pd.get_dummies(df['weather_main'])

    features = pd.DataFrame({
        'temp': df['temp'],
        'rain_1h': df['rain_1h'],
        'snow_1h': df['snow_1h'],
        'clouds_all': df['clouds_all'],
        'month': months,
        'day_of_month': day_of_month,
        'hour': hours,
        'is_holiday': is_holiday,
        'is_weekday': is_weekday
    })
    features = pd.concat([features, week, daypart, weathers], axis=1)
    y = df['traffic_volume']
    return features, y


def build_model(X, y, mothed):
    _name = re.findall(r'(\w+)', mothed.__str__())[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    time_a = time.time()
    mothed.fit(X_train, y_train)
    y_pred = mothed.predict(X_test)
    time_b = time.time()
    print(f'{"-" * 10}{_name}{"-" * 10}')
    print(f'R2_score: {round(metrics.r2_score(y_test, y_pred) * 100, 2)}%')
    print(f'time:{round(time_b - time_a, 3)}秒')
    return (X, y_pred, y_test, _name)


def visualization(X, y_pred, y_test, _name):
    _len = -400
    index_ordered = [f'{x}月{y}日{t}点' for x, y, t in zip(X.month, X.day_of_month, X.hour)][_len:]
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(16, 9))
    plt.plot(index_ordered, y_test[_len:], color='k', ls='-', label='actual')
    plt.plot(index_ordered, y_pred[_len:], color='b', ls='--', label='predicted; with date-time features')
    plt.xticks(index_ordered[::20], rotation=90)
    plt.legend()
    plt.title(f'{_name} predicted curve')
    plt.show()


def main():
    df = read_data()
    features, y = characteristics_the_engineer(df)
    params = {
        'n_estimators': 500,  # 决策树个数
        'max_depth': 4,
        'min_samples_split': 5,
        'learning_rate': 0.01,
        'loss': 'ls'
    }
    gb_reg = ensemble.GradientBoostingRegressor(**params)
    lg_reg = LGBMRegressor(boosting_type='gbdt', n_estimators=500)
    xg_reg = XGBRegressor(
        n_estimators=500,
        max_depth=15,
        learning_rate=0.01,
    )
    for i in [gb_reg, xg_reg, lg_reg]:
        X, y_pred, y_test, _name = build_model(features, y, i)
        visualization(X, y_pred, y_test, _name)

    # # <editor-fold desc="GridSearchCV调参，设备跑不动">
    # param_grid={
    #     'max_depth':[4,5,6],
    #     'n_estimators':[30,50,70],
    #     'learning_rate':[0.01,0.02,0.03]
    # }
    # xg_reg_adjustable_parameter=XGBRegressor()
    # xg_cv=GridSearchCV(estimator=xg_reg_adjustable_parameter,param_grid=param_grid,scoring='r2',cv=5)
    # X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.1, shuffle=False)
    # xg_cv.fit(X_train,y_train)
    # y_pred=xg_cv.predict(X_test)
    # print(f'r2 score:{metrics.r2_score(y_test,y_pred)}')
    # print(f'最佳参数：{xg_cv.best_params_}')
    # # </editor-fold>




if __name__ == '__main__':
    main()
