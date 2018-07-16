import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def pre_data():
    data = pd.read_csv('../data_2g.csv', index_col=None)
    param = pd.read_csv('../2g_gongcan.csv', index_col=None)
    # merge
    for i in range(1, 8):
        gongcan_temp = param.rename(columns={'RNCID': 'RNCID_' + str(i), 'CellID': 'CellID_' + str(i), 'Latitude': 'Latitude_' + str(i), 'Longitude': 'Longitude_' + str(i)})
        data = pd.merge(data, gongcan_temp, how='left', on=['RNCID_' + str(i), 'CellID_' + str(i)])
    
    # make groups
    groups = []
    for index, row in param.rename(columns={'RNCID': 'RNCID_1', 'CellID': 'CellID_1'}).iterrows():
        group = pd.merge(data, pd.DataFrame().append(row).drop(['Latitude', 'Longitude'], axis=1), on=['RNCID_1', 'CellID_1'])
        if group.size > 1:
            groups.append(group)

    return groups

def reverse(pred, test_set):
    result = pd.DataFrame(data=pred, columns=['pred_x', 'pred_y'])
    result['Longitude'] = result['pred_x'] / 100000000.0+ test_set.iloc[0]['Longitude_1']
    result['Latitude'] = result['pred_y'] / 100000000.0 + test_set.iloc[0]['Latitude_1']

    return result.drop(['pred_x', 'pred_y'], axis = 1)

def initDataSet(data):
    # 训练集
    training_set_temp = data.sample(
        frac=0.8 if data.shape[0] > 2 else 0.5).sort_index()

    training_labels = pd.DataFrame()
    training_labels['relative_x'] = training_set_temp['relative_x'] * 100000000.0
    training_labels['relative_y'] = training_set_temp['relative_y'] * 100000000.0
    training_labels['relative_x'] = training_labels['relative_x'].apply(int)
    training_labels['relative_y'] = training_labels['relative_y'].apply(int)
    training_set = training_set_temp.drop(['IMSI','Longitude','Latitude', 'Latitude_2' ,'Longitude_2', 'Latitude_3', 'Longitude_3',
                                        'Latitude_4' ,'Longitude_4' ,'Latitude_5' ,'Longitude_5' ,'Latitude_6',
                                        'Longitude_6' ,'Latitude_7', 'Longitude_7', 'relative_x' ,'relative_y'], axis=1).fillna(0)

    # 测试集
    test_set_temp = data.append(training_set_temp).drop_duplicates(keep=False).sort_index()
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

    return training_set, training_labels, test_set, test_labels, test_pos

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

    persents = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    plt.title("CDF Figure")
    plt.xlabel("persents")
    plt.ylabel('Error(meters)')

    ave_precision_x = []
    ave_precision_y = []
    ave_r_x = []
    ave_r_y = []
    ave_f_x = []
    ave_f_y = []
    for group in data_set:
        score_set = []
        group['relative_x'] = group['Longitude'] - group['Longitude_1']
        group['relative_y'] = group['Latitude'] - group['Latitude_1']
        precision_x = []
        precision_y = []
        r_x = []
        r_y = []
        f1_x = []
        f1_y = []
        for i in range(0, 10):
            training_set, training_labels, test_set, test_labels, test_pos = initDataSet(group)
            # RandomForestClassifier
            # 训练
            clf = RandomForestClassifier().fit(training_set, training_labels)
            # 测试
            pred = clf.predict(test_set)
            # 转换
            result = reverse(pred, test_set)
            # 评估
            deviation = evaluate(test_pos, result)
            score_set.append(deviation)
            #precision
            p_score_lon = precision_score(test_labels['relative_x'], pd.DataFrame(data=pred)[0], average='macro')
            p_score_lat = precision_score(test_labels['relative_y'], pd.DataFrame(data=pred)[1], average='macro')
            precision_x.append(p_score_lon)
            precision_y.append(p_score_lat)
            
            #recall
            r_score_lon = recall_score(test_labels['relative_x'], pd.DataFrame(data=pred)[0], average='macro')
            r_score_lat = recall_score(test_labels['relative_y'], pd.DataFrame(data=pred)[1], average='macro')
            r_x.append(r_score_lon)
            r_y.append(r_score_lat)

            #f1_score
            f1_score_lon = f1_score(test_labels['relative_x'], pd.DataFrame(data=pred)[0], average='macro')
            f1_score_lat = f1_score(test_labels['relative_y'], pd.DataFrame(data=pred)[1], average='macro')
            f1_x.append(f1_score_lon)
            f1_y.append(f1_score_lat)

        score_set = pd.DataFrame(data=score_set)
        average_score = []
        for i in range(0, 10):
            average_score.append(score_set[i].mean())
        ave_precision_x.append(sum(precision_x)/len(precision_x))
        ave_precision_y.append(sum(precision_y)/len(precision_y))
        ave_r_x.append(sum(r_x)/len(r_x))
        ave_r_y.append(sum(r_y)/len(r_y))
        ave_f_x.append(sum(f1_x)/len(f1_y))
        ave_f_y.append(sum(f1_y)/len(f1_y))

        # #画图
        # persents = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # plt.title("CDF Figure")
        # plt.xlabel("persents")
        # plt.ylabel('Error(meters)')
        plt.plot(persents, average_score, 'r*-')
    for i in range(0, len(data_set)):
        print('group_' + str(i) + ':')
        print('percision: relative_x:', ave_precision_x[i], 'relative_y:', ave_precision_y[i])
        print('recall: relative_x:', ave_r_x[i], 'relative_y:', ave_r_y[i])
        print('f1_score:relative_x:', ave_f_x[i], 'relative_y:', ave_f_y[i])

    plt.legend(bbox_to_anchor=[0.3, 1])
    plt.grid()
    plt.show()
