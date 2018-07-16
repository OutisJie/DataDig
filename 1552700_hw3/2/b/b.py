import pandas as pd
import numpy as np
import time

from scipy.optimize import leastsq
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn import metrics

def pre_data():
    data = pd.read_csv('../output_bak.csv', header=0)
    
    data = data.drop(['Unnamed: 0', 'uid', 'sldatime', 'pno', 'cno', 'cmrid', 'id',
       'bcd', 'pluname', 'spec', 'pkunit', 'dptname', 'bndname', 'qty', 'disamt', 
       'ismmx', 'mtype', 'mdocno','isdel', 'date'], axis = 1)

    return data

def init_data(data, grouped):
    keys = []
    for key, group in grouped:
        sets = []
        sets.append(key[0])
        sets.append(key[1])
        keys.append(sets)
    feature = pd.DataFrame(data=keys, columns=['vipno', 'pluno'] )
    # print(feature['vipno'].values)

    # 对每个组，key=(vipno, pluno), 是否存在5月份的数据
    ui_group_dic = []
    for key, group in grouped:
        if 5 in group['month'].tolist():
            ui_group_dic.append( True)
        else:
            ui_group_dic.append( False)
    new_col = pd.DataFrame(data=ui_group_dic, columns = ['label'])
    feature = pd.concat([feature, new_col], axis =1)
    months = [2, 3, 4]
    train_feature = get_features(data, grouped, feature, months)
    months = [5, 6, 7]
    test_feature = get_features(data, grouped, feature, months)
    test_feature = test_feature.drop(['label'], axis = 1)
    
    training_labels = train_feature['label'].values
    training_set = train_feature.drop(['label'], axis = 1)
    return training_set, training_labels, test_feature

def get_features(data, grouped, feature, months):
    # 对每个组中的每一个特征,判断它和key是否为一对一关系
    # 如： key - ui_count_month是一对多
    # 如： key - ui_count_whole是一对一
    col_name = data.columns.values
    for i in range(6, 198):
        if col_name[i].count('_whole') > 0:
            new_col = []
            for key, group in grouped:
                #print(group)
                new_col.append(group[col_name[i]].values[0])
            new_col = pd.DataFrame(data=new_col, columns = [col_name[i]])
            feature = pd.concat([feature, new_col], axis = 1)
        elif col_name[i].count('_month') > 0:
            new_cols = [[], [], []]
            new_cols_last = [[], [], []]
            for key, group in grouped:
                # 这个组里面的所有month-col_name的对应值
                for index, row in group.iterrows():
                    if row['month'] == months[0]:
                        new_cols[0].append(row[col_name[i]])
                    elif row['month'] == months[1]:
                        new_cols[1].append(row[col_name[i]])
                    elif row['month'] == months[2]:
                        new_cols[2].append(row[col_name[i]])
                # 这时候有多个2，多个3，多个4，只取其中一个
                # 也可能都没有，补0
                #print(new_cols)
                for index in range(0, 3):
                    if len(new_cols[index]) is not 0:
                        new_cols_last[index].append(new_cols[index][0])
                    else:
                        new_cols_last[index].append(0)
            new_feature = []
            for j in range(0, feature.shape[0]):
                new_feature.append([new_cols_last[0][j], new_cols_last[1][j] ,new_cols_last[2][j]])
            # print(new_feature)
            new_cols_last = pd.DataFrame(data=new_feature , columns = ['1_' + col_name[i], '2_' + col_name[i], '3_' + col_name[i]])
            feature = pd.concat([feature, new_cols_last], axis = 1)

    print(feature)
    return feature

def save_result(test_features, pre_data, classifier_name):
    with open( 'result/' + classifier_name + '_predict.txt', 'a+') as f:
        content = 'vipno, pluno, if buy\n' 
        for index in range(0, len(pre_data)):
            content = content + str(test_features['vipno'].values[index]) + ',' + str(test_features['pluno'].values[index]) + ':' + str(pre_data[index]) + '\n'

        f.write(content)

def make_time_picture(times, labels):
    plt.figure(figsize=(15, 8))
    plt.xlabel('classifier')
    plt.ylabel('cost(s)')
    plt.title('Time performance bar diagram')
    index = np.arange(6)
    bar_width = 0.2
    plt.bar(index, times, bar_width, color='g', label='old_data')
    plt.xticks([x + 0.2 for x in index], ['GaussianNB', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'AdaBoostClassifier',
              'BaggingClassifier', 'RandomForestClassifier'])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data = pre_data()

    ui_grouped = data.groupby(['vipno', 'pluno'])
    
    # get train data
    train_set, train_lables, test_set = init_data(data, ui_grouped)
    
    #初始化
    classifiers = [GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier(), AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
                 ,BaggingClassifier(base_estimator=KNeighborsClassifier()), RandomForestClassifier(), GradientBoostingClassifier(n_estimators=50)]
    labels = ['GaussianNB', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'AdaBoostClassifier',
              'BaggingClassifier', 'RandomForestClassifier'  , 'GradientBoostingClassifier']
    
    scores = []
    # precisions = [[], [], [], [], [], [], []]
    # recalls = [[], [], [], [], [], [], []]
    # f1_scores = [[], [], [], [], [], [], []]
    times = []

    for index in range(0, 6):
        start = time.time()
        clf = classifiers[index]
        # 训练
        clf = clf.fit(train_set, train_lables)

        pred = clf.predict(test_set)

        end = time.time()
        times.append(round(end - start, 2))

        print(times[index], 's', labels[index] )
        # write
        save_result(test_set, pred, labels[index])

    make_time_picture(times, labels)
