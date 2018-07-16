import numpy as np
import pandas as pd

import matplotlib.pyplot as plt  # 绘图库
from scipy.optimize import leastsq  # 引入最小二乘法算法

##需要拟合的函数func :指定函数的形状
def func(p, x):
    k, b = p
    return k*x+b

##偏差函数：x,y都是列表:这里的x,y更上面的Xi,Yi中是一一对应的
def error(p, x, y):
    return func(p, x)-y

def get_trend(grouped, feature):
    trend = {}
    error_set = {}

    for key, group in grouped:
        Y = []
        for month in ['02', '03', '04', '05', '06', '07']:
            if group[group['month'] == month].size > 0:
                Y.append(group[group['month'] == month].iloc[0][feature])
            else:
                Y.append(0)

        # trend
        X = np.array([1, 2, 3, 4, 5, 6])
        Y = np.array(Y)
        #k,b的初始值，可以任意设定,经过几次试验，发现p0的值会影响cost的值：Para[1]
        p0 = [1, 20]
        #把error函数中除了p0以外的参数打包到args中(使用要求)
        Para = leastsq(error, p0, args=(X, Y))
        #读取结果
        k, b = Para[0]

        # #画样本点
        # plt.figure(figsize=(8,6)) ##指定图像比例： 8：6
        # plt.scatter(X,Y,color="green",label="样本数据",linewidth=2)

        # #画拟合直线
        # x=np.linspace(0,12,100) ##在0-15直接画100个连续点
        # y=k*x+b ##函数式
        # plt.plot(x,y,color="red",label="拟合直线",linewidth=2)
        # plt.legend(loc='lower right') #绘制图例
        # plt.show()

        trend[key] = k
        temp = 0
        for i in range(0, 5):
            temp += abs(Y[5]-Y[i])
        error_set[key] = temp/5
    
    return trend, error_set


def type4_type1_trend(data_set, feild, label):
    for feature in [label + '_count_month', label + '_money_month', label + '_days_month']:
        grouped = data_set[[feild, 'month', feature]].groupby([feild])
        trend, error_set = get_trend(grouped, feature)
        trends = []
        errors = []
        for index, row in data_set.iterrows():
            trends.append(trend[row[feild]])
            errors.append(error_set[row[feild]])
        data_set = pd.concat([data_set, pd.DataFrame(data=trends, columns=[feature + '_trend'])], axis=1)
        data_set = pd.concat([data_set, pd.DataFrame(data=errors, columns=[feature + '_error'])], axis=1)

    return data_set


def type4_type1_trend_tw(data_set, feild1, feild2, label):
    for feature in [label + '_count_month', label + '_money_month', label + '_days_month']:
        grouped = data_set[[feild1, feild2, 'month', feature]].groupby([feild1, feild2])
        trend, error_set = get_trend(grouped, feature)

        trends = []
        errors = []
        for index, row in data_set.iterrows():
            trends.append(trend[(row[feild1], row[feild2])])
            errors.append(error_set[(row[feild1], row[feild2])])
        data_set = pd.concat([data_set, pd.DataFrame(data=trends, columns=[feature + '_trend'])], axis=1)
        data_set = pd.concat([data_set, pd.DataFrame(data=errors, columns=[feature + '_error'])], axis=1)

    return data_set

def type4_type1_trend_pd(data_set, feild1, feild2, label):
    feature = label + '_count_month'
    grouped = data_set[[feild1, feild2, 'month', feature]].groupby([feild1, feild2])
    trend, error_set = get_trend(grouped, feature)

    trends = []
    errors = []
    for index, row in data_set.iterrows():
        trends.append(trend[(row[feild1], row[feild2])])
        errors.append(error_set[(row[feild1], row[feild2])])
    data_set = pd.concat([data_set, pd.DataFrame(data=trends, columns=[feature + '_trend'])], axis=1)
    data_set = pd.concat([data_set, pd.DataFrame(data=errors, columns=[feature + '_error'])], axis=1)

    return data_set

def type4_repeat(feild1, feild2, data_set):
    # count
    once = data_set.drop_duplicates(subset=[feild1, feild2],keep='first')
    much = data_set.drop_duplicates(subset=[feild1, feild2],keep=False)  
    much = once.append(much).drop_duplicates(subset=[feild1, feild2],keep=False)  

    count = much.drop_duplicates(subset=[feild1, feild2],keep='first')
    count = count[feild2].groupby(count[feild1]).count()
    whole = once[feild2].groupby(data_set[feild1]).count()

    new_col1 = []
    new_col2 = []
    for index, row in data_set.iterrows():
        if row[feild1] in count.index:
            new_col1.append(count.loc[row[feild1]])
            new_col2.append(count.loc[row[feild1]]/whole.loc[row[feild1]])
        else:
            new_col1.append(0)
            new_col2.append(0)

    new_col1 = pd.DataFrame(data=new_col1, columns=[feild1 + '_' + feild2 + '_repeat_buy_count'])
    new_col2 = pd.DataFrame(data=new_col2, columns=[feild1 + '_' + feild2 + '_buy_ratio'])
    data_set = pd.concat([data_set, new_col1], axis=1)
    data_set = pd.concat([data_set, new_col2], axis=1)
    
    # day
    once = data_set.drop_duplicates(subset=[feild1, feild2, 'date'],keep='first')
    much = data_set.drop_duplicates(subset=[feild1, feild2, 'date'],keep=False)  
    much = once.append(much).drop_duplicates(subset=[feild1, feild2, 'date'],keep=False)  

    count = much[feild2].groupby(much[feild1]).count()
    whole = data_set[feild2].groupby(data_set[feild1]).count()

    new_col1 = []
    new_col2 = []
    for index, row in data_set.iterrows():
        if row[feild1] in count.index:
            new_col1.append(count.loc[row[feild1]])
            new_col2.append(count.loc[row[feild1]]/whole.loc[row[feild1]])
        else:
            new_col1.append(0)
            new_col2.append(0)

    new_col1 = pd.DataFrame(data=new_col1, columns=[feild1 + '_' + feild2 + '_repeat_buy_day'])
    new_col2 = pd.DataFrame(data=new_col2, columns=[feild1 + '_' + feild2 + '_day_ratio'])

    data_set = pd.concat([data_set, new_col1], axis=1)
    data_set = pd.concat([data_set, new_col2], axis=1)
    
    return data_set


def type4(data):
    #------------trend,平均差---------------------
    data_set = type4_type1_trend(data, 'vipno', 'user')
    data_set = type4_type1_trend(data_set, 'bndno', 'brand')
    data_set = type4_type1_trend(data_set, 'dptno', 'cate')
    data_set = type4_type1_trend(data_set, 'pluno', 'item')

    data_set = type4_type1_trend_tw(data_set, 'vipno', 'bndno', 'ub')
    data_set = type4_type1_trend_tw(data_set, 'vipno', 'dptno', 'uc')
    data_set = type4_type1_trend_tw(data_set, 'vipno', 'pluno', 'ui')
    data_set = type4_type1_trend_tw(data_set, 'bndno', 'dptno', 'bc')

    data_set = type4_type1_trend_pd(data_set, 'vipno', 'pluno', 'ui_unique')
    data_set = type4_type1_trend_pd(data_set, 'vipno', 'bndno', 'ub_unique')
    data_set = type4_type1_trend_pd(data_set, 'vipno', 'dptno', 'uc_unique')
    data_set = type4_type1_trend_pd(data_set, 'bndno', 'pluno', 'bi_unique')
    data_set = type4_type1_trend_pd(data_set, 'dptno', 'pluno', 'ci_unique')

    data_set = type4_type1_trend_pd(data_set, 'bndno', 'vipno', 'bu_people')
    data_set = type4_type1_trend_pd(data_set, 'dptno', 'vipno', 'cu_people')
    data_set = type4_type1_trend_pd(data_set, 'pluno', 'vipno', 'iu_people')
    #-----------repeat feature------------------
    # user
    data_set = type4_repeat('vipno', 'bndno', data_set)
    data_set = type4_repeat('vipno', 'dptno', data_set)
    data_set = type4_repeat('vipno', 'pluno', data_set)

    return data_set
