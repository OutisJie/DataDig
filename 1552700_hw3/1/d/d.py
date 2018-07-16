import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time

def pre_data():
    data = pd.read_csv('../data_2g.csv', index_col=None)
    param = pd.read_csv('../2g_gongcan.csv', index_col=None)
    # merge
    for i in range(1, 8):
        gongcan_temp = param.rename(columns={'RNCID': 'RNCID_' + str(i), 'CellID': 'CellID_' + str(
            i), 'Latitude': 'Latitude_' + str(i), 'Longitude': 'Longitude_' + str(i)})
        data = pd.merge(data, gongcan_temp, how='left', on=[
                        'RNCID_' + str(i), 'CellID_' + str(i)])
    # make groups
    groups = []
    for index, row in param.rename(columns={'RNCID': 'RNCID_1', 'CellID': 'CellID_1'}).iterrows():
        group = pd.merge(data, pd.DataFrame().append(row).drop(
            ['Latitude', 'Longitude'], axis=1), on=['RNCID_1', 'CellID_1'])
        if group.size > 1:
            groups.append(group)

    return groups


def relative_pos(group):
    group['relative_x'] = group['Longitude'] - group['Longitude_1']
    group['relative_y'] = group['Latitude'] - group['Latitude_1']

    return group

def reverse(result, test_set):
    pos_set = pd.DataFrame()
    pos_set['Longitude_1'] = test_set['Longitude_1']
    pos_set['Latitude_1'] = test_set['Latitude_1']
    pos_set = pos_set.reset_index() 
    result = pd.concat([result, pos_set], axis=1, ignore_index=True)
    result.columns = ['pred_x','pred_y','no','Longitude_1','Latitude_1']
    
    result['Longitude'] = result['pred_x'] / 100000000.0+ result['Longitude_1']
    result['Latitude'] = result['pred_y'] / 100000000.0 + result['Latitude_1']

    return result.drop(['pred_x','pred_y','no','Longitude_1','Latitude_1'], axis = 1)

def initDataSet(data):
    # 训练集
    train_set_temp = data.sample(
        frac=0.8 if data.shape[0] > 2 else 0.5).sort_index()

    train_labels = pd.DataFrame()
    train_labels['relative_x'] = train_set_temp['relative_x'] * 100000000.0
    train_labels['relative_y'] = train_set_temp['relative_y'] * 100000000.0
    train_labels['relative_x'] = train_labels['relative_x'].apply(int)
    train_labels['relative_y'] = train_labels['relative_y'].apply(int)
    train_set = train_set_temp.drop(['IMSI' ,'Longitude','Latitude', 'Latitude_2' ,'Longitude_2', 'Latitude_3', 'Longitude_3',
                                        'Latitude_4' ,'Longitude_4' ,'Latitude_5' ,'Longitude_5' ,'Latitude_6',
                                        'Longitude_6' ,'Latitude_7', 'Longitude_7', 'relative_x' ,'relative_y'], axis=1).fillna(0)

    # 测试集
    test_set_temp = data.append(train_set_temp).drop_duplicates(keep=False).sort_index()
    test_labels = pd.DataFrame()
    test_labels['relative_x'] = test_set_temp['relative_x'] * 100000000.0
    test_labels['relative_y'] = test_set_temp['relative_y'] * 100000000.0
    test_labels['relative_x'] = test_labels['relative_x'].apply(int)
    test_labels['relative_y'] = test_labels['relative_y'].apply(int)
    test_set = test_set_temp.drop(['IMSI','Longitude','Latitude', 'Latitude_2' ,'Longitude_2', 'Latitude_3', 'Longitude_3',
                                        'Latitude_4' ,'Longitude_4' ,'Latitude_5' ,'Longitude_5' ,'Latitude_6',
                                        'Longitude_6' ,'Latitude_7', 'Longitude_7', 'relative_x' ,'relative_y'], axis=1).fillna(0)

    test_pos = pd.DataFrame()
    test_pos['Longitude'] = test_set_temp['relative_x'] + test_set_temp['Longitude_1']
    test_pos['Latitude'] = test_set_temp['relative_y'] + test_set_temp['Latitude_1']

    return train_set, train_labels, test_set, test_labels, test_pos

def evaluate(test_pos, result):
    compare_set = pd.concat([test_pos.reset_index(drop=True), result.reset_index(drop=True)], axis=1, ignore_index=True)
    compare_set['lon_dev'] = abs(compare_set[0] - compare_set[2])
    compare_set['lat_dev'] = abs(compare_set[1] - compare_set[3])
    compare_set['error'] = compare_set['lon_dev'] + compare_set['lat_dev']
    compare_set = compare_set.sort_values(['error'])

    deviation = [0]
    for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        deviation.append(haversine(compare_set.iloc[int(i*compare_set.shape[0]), ]['error'], 0, 0, 0))

    return deviation

def haversine(lon1, lat1, lon2, lat2):
    """ 
    Calculate the great circle distance between two points  
    on the earth (specified in decimal degrees) 
    """
    # 将十进制转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

if __name__ == '__main__':
    data_set = pre_data()
    median_score = []

    for group in data_set:
        score_set = []
        group_set = relative_pos(group)
        # print(group_set[['relative_x', 'relative_y']])

        for i in range(0, 10):
            train_set, train_labels, test_set, test_labels, test_pos = initDataSet(group_set)
            # RandomForestClassifier
            # 训练
            clf = RandomForestClassifier().fit(train_set, train_labels)
            # 测试
            pred = clf.predict(test_set)
            # 转换
            result = reverse(pd.DataFrame(data=pred, columns=['pred_x', 'pred_y']), test_set)
            # 评估
            deviation = evaluate(test_pos, result)
            score_set.append(deviation)
           
        score_set = pd.DataFrame(data=score_set)
        average_score = []
        for i in range(0, 10):
            average_score.append(score_set[i].mean()) 
        #print(average_score)
        median_score.append(average_score[5])
        #print(median_score)

    # topK
    K = int(0.2 * len(median_score))
    # 前K个
    topk_median_score_plus = np.sort(median_score)[:K]
    # 后K个
    topk_median_score_minus = np.sort(median_score)[-K:]

    # 重新分组
    topk_plus = []
    topk_minus = []
    for i in range(0, len(median_score)):
        if median_score[i] in topk_median_score_plus:
            topk_plus.append(data_set[i])
        elif median_score[i] in topk_median_score_minus:
            topk_minus.append(data_set[i])
    
    plt.title("CDF Figure")
    plt.xlabel("persents")
    plt.ylabel('Error(meters)')
    time_old = []
    time_new = []
    # 重新训练
    for k in range(0, K):
        persents = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # 对于原来的minus的group
        old_group = topk_minus[k]
        old_score_set= []
        old_group_set = relative_pos(old_group)
        exec_time = []
        # 训练
        for i in range(0, 10):
            old_train_set, old_train_labels, old_test_set, old_test_labels, old_test_pos = initDataSet(old_group_set)
            start = time.time()
            old_clf = RandomForestClassifier().fit(old_train_set, old_train_labels)
            old_pred = old_clf.predict(old_test_set)
            end = time.time()
            exec_time.append(round(end - start, 2))
            old_result = reverse(pd.DataFrame(data=old_pred, columns=['pred_x', 'pred_y']), old_test_set)
            old_deviation = evaluate(old_test_pos, old_result)
            old_score_set.append(old_deviation)
        
        time_old.append(sum(exec_time)/len(exec_time))
        # 旧结果
        old_average_score = []
        old_score_set = pd.DataFrame(data=old_score_set)
        for i in range(0, 10):
            old_average_score.append(old_score_set[i].mean())
        # 画图
        plt.plot(persents, old_average_score, 'r-', label= 'group_' + str(k) + 'old_data' )

        # 对于加入了plus数据的新group
        # new_group = topk_minus[k]
        # for topk in topk_plus:
        #     new_group = pd.concat([new_group, topk])
        new_group = pd.concat([topk_minus[k], topk_plus[k]])
        new_score_set= []
        new_group_set = relative_pos(new_group)
        exec_time = []
        # 训练
        for i in range(0, 10):
            new_train_set, new_train_labels, new_test_set, new_test_labels, new_test_pos = initDataSet(new_group_set)
            start = time.time()
            new_clf = RandomForestClassifier().fit(new_train_set, new_train_labels)
            new_pred = new_clf.predict(new_test_set)
            end = time.time()
            exec_time.append(round(end - start, 2))
            new_result = reverse(pd.DataFrame(data=new_pred, columns=['pred_x', 'pred_y']), new_test_set)
            new_deviation = evaluate(new_test_pos, new_result)
            new_score_set.append(new_deviation)
        time_new.append(sum(exec_time)/len(exec_time))
        # 新结果
        new_average_score = []
        new_score_set = pd.DataFrame(data=new_score_set)
        for i in range(0, 10):
            new_average_score.append(new_score_set[i].mean())
        # 画图
        plt.plot(persents, new_average_score, 'b*-', label= 'group_' + str(k) + 'new_data')

    plt.legend(bbox_to_anchor=[0.3, 1])
    plt.grid()
    plt.show()

    # 时间图
    plt.xlabel('group')
    plt.ylabel('cost(s)')
    plt.title('Time performance bar diagram')
    index = np.arange(8)
    bar_width = 0.2
    plt.bar(index, time_old, bar_width, color='r', label='old_data')
    plt.bar(index+bar_width, time_new, bar_width, color='b', label='new_data')
    plt.xticks([x + 0.2 for x in index], ['group1', 'group2', 'group3', 'group4', 'group5', 'group6', 'group7', 'group8' ])
    plt.legend()
    plt.show()
