import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time

def haversine(lon1, lat1, lon2, lat2):
    """ 
    Calculate the great circle distance between two points  
    on the earth (specified in decimal degrees) 
    """
    # 将十进制转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = abs(lon2 - lon1)
    dlat = abs(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

def pre_data(lon1, lat1, grid_x_num, grid_lon_len, grid_lat_len):
    data = pd.read_csv('../data_2g.csv', index_col=None)
    param = pd.read_csv('../2g_gongcan.csv', index_col=None)
    # merge
    for i in range(1, 8):
        param_temp = param.rename(columns={'RNCID': 'RNCID_' + str(i), 'CellID': 'CellID_' + str(i), 'Latitude': 'Latitude_' + str(i), 'Longitude': 'Longitude_' + str(i)})
        data = pd.merge(data, param_temp, how='left', on=['RNCID_' + str(i), 'CellID_' + str(i)])

    # 给每条数据添加Grid ID
    data['grid_x'] = (data['Longitude'] - lon1)/grid_lon_len
    data['grid_y'] = (data['Latitude'] - lat1)/grid_lat_len
    data['grid_x'] = data['grid_x'].apply(int)
    data['grid_y'] = data['grid_y'].apply(int)
    data['grid'] = data['grid_x'] + data['grid_y'] * grid_x_num
    
    return data

def reverse(pred, lon1, lat1, grid_x_num, grid_lon_len, grid_lat_len):
    result = pd.DataFrame(data=pred, columns=['grid'])
    # 把grid ID 转换成栅格中心点的经纬度
    result['grid_y'] = result['grid']/grid_x_num
    result['grid_y'] = result['grid_y'].apply(int)
    result['grid_x'] = result['grid'] - result['grid_y'] * grid_x_num
    result['Longitude_center'] = (result['grid_x'] + 0.5) * grid_lon_len + lon1
    result['Latitude_center'] = (result['grid_y'] + 0.5) * grid_lat_len + lat1
    return result.drop(['grid_x', 'grid_y'], axis=1)

def initDataSet(data):
    # 训练集
    training_set_temp = data.sample(frac=0.8)
    training_labels = training_set_temp['grid']
    training_set = training_set_temp.drop(['IMSI', 'Longitude','Latitude', 'grid_x', 'grid_y', 'grid', 'Latitude_1' ,'Longitude_1' ,'Latitude_2', 'Longitude_2',
                                        'Latitude_3', 'Longitude_3', 'Latitude_4', 'Longitude_4' ,'Latitude_5',
                                        'Longitude_5', 'Latitude_6', 'Longitude_6' ,'Latitude_7', 'Longitude_7'], axis=1).fillna(0)

    # 测试集
    test_set_temp = data.append(training_set_temp).drop_duplicates(keep=False)
    test_labels = test_set_temp['grid'].values
    test_pos = pd.DataFrame()
    test_pos['Longitude'] = test_set_temp['Longitude']
    test_pos['Latitude'] = test_set_temp['Latitude']
    IMSI = test_set_temp['IMSI']
    test_set = test_set_temp.drop(['IMSI', 'Longitude','Latitude', 'grid_x', 'grid_y', 'grid', 'Latitude_1' ,'Longitude_1' ,'Latitude_2', 'Longitude_2',
                                        'Latitude_3', 'Longitude_3', 'Latitude_4', 'Longitude_4' ,'Latitude_5',
                                        'Longitude_5', 'Latitude_6', 'Longitude_6' ,'Latitude_7', 'Longitude_7'], axis=1).fillna(0)

    
    return training_set, test_set, training_labels, test_labels, test_pos, IMSI


def evaluate(test_set, result_set):
    # 计算
    compare_set = pd.concat([test_set.reset_index(drop=True), result_set.reset_index(drop=True)], axis=1, ignore_index=True)
    compare_set['lon_dev'] = abs(compare_set[0] - compare_set[3])
    compare_set['lat_dev'] = abs(compare_set[1] - compare_set[4])

    # 计算准确率
    compare_set['error'] = compare_set['lon_dev'] + compare_set['lat_dev']
    compare_set = compare_set.sort_values(['error'])

    deviation = [0]
    for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        deviation.append(haversine(compare_set.iloc[int(
            i*compare_set.shape[0]), ]['error'], 0, 0, 0))

    return deviation

def revise(IMSI, test_set, test_pos, result):
    test_set = test_set.reset_index(drop=True)
    points = pd.concat([result, test_pos.reset_index(drop=True)], axis= 1)
    points['IMSI'] = IMSI.reset_index(drop=True)
    points['MRTime'] = test_set['MRTime']
    grouped = points.groupby(['IMSI'])
    for key, group in grouped:  
        group = group.sort_values(by = 'MRTime',axis = 0,ascending = True)
        start = 0 #异常的起始点
        end = 0
        entry = False #异常入口
        totol_time = 0
        for i in range(1, group.shape[0]):
            point1 = group.iloc[i-1, ]
            point2 = group.iloc[i, ]
            distance = haversine(point1.loc['Longitude_center'], point1.loc['Latitude_center'], point2.loc['Longitude_center'], point2.loc['Latitude_center'])
            time = abs(point1.loc['MRTime'] - point2.loc['MRTime']) / 1000
            speed = distance / time
            if entry is True:
                point1 = group.iloc[start - 1, ] # 异常点之前的正常点
                point2 = group.iloc[i, ]
                distance = haversine(point1.loc['Longitude_center'], point1.loc['Latitude_center'], point2.loc['Longitude_center'], point2.loc['Latitude_center'])
                totol_time += time
                speed = distance / totol_time
            if speed > 10 and entry is False: #异常点入口
                start = i
                totol_time += time
                entry = True
            if speed < 10 and entry is True:
                entry = False
                end = i # 从start到end都是异常点
                # 修改grid_id,改为start - 1所在的grid
                keys = []
                id = group.iloc[start - 1, ].loc['grid']
                lon = group.iloc[start - 1, ].loc['Longitude_center']
                lat = group.iloc[start - 1, ].loc['Latitude_center']
                for index in range(start, end):
                    keys.append( group.iloc[index, ].loc['grid'])
                for j, row in result.iterrows():
                    if row.loc['grid'] in keys:
                        row.loc['grid'] = id
                        row.loc['Longitude_center'] = lon
                        row.loc['Latitude_center'] = lat
                start = 0 #异常的起始点
                end = 0
                totol_time = 0
    return result

def get_sum(temp):
    sum = 0
    for i in range(0 ,len(temp)):
        sum += temp[i]
    return sum

if __name__ == '__main__':
    # 参数
    lon1 = 121.20120490000001
    lat1 = 31.28175691
    lon2 = 121.2183295
    lat2 = 31.29339344
    width = haversine(lon1, lat1, lon2, lat1)
    height = haversine(lon1, lat1, lon1, lat2)
    grid_edge = 20
    # 每个栅格的个数
    grid_x_num = int(width/grid_edge)
    grid_y_num = int(height/grid_edge)
    # 每个栅格的经纬宽度
    grid_lon_len = (lon2 - lon1)/grid_x_num
    grid_lat_len = (lat2 - lat1)/grid_y_num

    #初始化
    data_set = pre_data(lon1, lat1, grid_x_num, grid_lon_len, grid_lat_len)

    classifiers = [GaussianNB(), KNeighborsClassifier(n_neighbors=4, weights='distance'), DecisionTreeClassifier(random_state=0),
                AdaBoostClassifier(base_estimator=DecisionTreeClassifier()), BaggingClassifier(), RandomForestClassifier(), GradientBoostingClassifier()]

    labels = ['GaussianNB', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'AdaBoostClassifier',
              'BaggingClassifier' , 'RandomForestClassifier', 'GradientBoostingClassifier']
    scores = [[], [], [], [], [], []]
    revise_scores = [[], [], [], [], [], []]
    precisions = [[], [], [], [], [], [], []]
    recalls = [[], [], [], [], [], [], []]
    f1_scores = [[], [], [], [], [], [], []]
    times = [[], [], [], [], [], [], []]

    for i in range(0, 10):
        # 整理训练集和测试集
        training_set, test_set, training_labels, test_labels, test_pos, IMSI = initDataSet(data_set)

        for index in range(0, 6):
            start = time.time()
            clf = classifiers[index]
            # 训练
            clf = clf.fit(training_set, training_labels)
            # 测试
            pred = clf.predict(test_set)
            # 将得到的gridID转换为经纬度
            result = reverse(pred, lon1, lat1, grid_x_num, grid_lon_len, grid_lat_len)
            # 判断偏差
            deviation = evaluate(test_pos, result)
            scores[index].append(deviation)

            revise_result = revise(IMSI, test_set, test_pos, result)
            revise_scores[index].append(evaluate(test_pos, revise_result))
            exec_time = time.time() - start
            times[index].append(round(exec_time, 2))

            #精确度precision
            precisions[index].append(precision_score(test_labels, pred, average='macro'))
            #Recall
            recalls[index].append(recall_score(test_labels, pred, average='macro'))
            # F-measurement
            f1_scores[index].append(f1_score(test_labels, pred, average='macro'))
            # 生成报告
            # print("GuassianNB report:", classification_report(test_labels, pred))
            # print(scores_GaussianNB)

    
    meters = []
    for score in scores:
        score = pd.DataFrame(data=score)
        average_score = []
        for i in range(0, 10):
            average_score.append(score[i].mean())
        meters.append(average_score)

    revise_meters = []
    for score in revise_scores:
        score = pd.DataFrame(data=score)
        average_score = []
        for i in range(0, 10):
            average_score.append(score[i].mean())
        revise_meters.append(average_score)
    # print(scores)
    # print(meters)
    # print(revise_meters)

    precision_set = []
    recall_set = []
    f1_set = []
    time_set = []
    for index in range(0, 6):
        precision_set.append(get_sum(precisions[index])/len(precisions[index]))
        recall_set.append(get_sum(recalls[index])/len(recalls[index]))
        f1_set.append(get_sum(f1_scores[index])/len(f1_scores[index]))
        time_set.append(get_sum(times[index])/len(times[index]))
        print(labels[index] + ', average_error :' , meters[index])
        print(labels[index] + ', average_precision:', precision_set[index])
        print(labels[index] + ', average_recall:' , recall_set[index])
        print(labels[index] + ', average_fi :' , f1_set[index])
        print(labels[index] + ', average_time :', time_set[index])

    # 绘制CDF曲线
    persents = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    plt.title('CDF Figure')
    plt.xlabel('persent')
    plt.ylabel('Error(meters)')
    # plt.plot(persents, meters[0], 'ro-', label=labels[0])
    # plt.plot(persents, meters[1], 'go-', label=labels[1])
    # plt.plot(persents, meters[2], 'bo-', label=labels[2])
    # plt.plot(persents, meters[3], 'yo-', label=labels[3])
    # plt.plot(persents, meters[4], 'mo-', label=labels[4])
    # plt.plot(persents, meters[5], 'co-', label=labels[5])
    # plt.plot(persents, meters[6], 'ko-', label=labels[6])
    plt.plot(persents, revise_meters[0], 'r*-', label=labels[0] + 'revise')
    plt.plot(persents, revise_meters[1], 'g*-', label=labels[1] + 'revise')
    plt.plot(persents, revise_meters[2], 'b*-', label=labels[2] + 'revise')
    plt.plot(persents, revise_meters[3], 'y*-', label=labels[3] + 'revise')
    plt.plot(persents, revise_meters[4], 'm*-', label=labels[4] + 'revise')
    plt.plot(persents, revise_meters[5], 'c*-', label=labels[5] + 'revise')
    # plt.plot(persents, revise_meters[6], 'k*-', label=labels[6] + 'revise')
    plt.legend(bbox_to_anchor=[0.3, 1])
    plt.grid()
    plt.show()

    # draw the score figure
    plt.title('Precision, Recall and F-measurement score Figure')
    plt.xlabel('Classifier')
    plt.ylabel('Score')
    index = np.arange(6)
    bar_width = 0.2
    plt.bar(index, precision_set, bar_width, color='r', label='Precision')
    plt.bar(index+bar_width, recall_set, bar_width, color='b', label='Recall')
    plt.bar(index+bar_width*2, f1_set, bar_width, color='g', label='F-measurement')
    plt.xticks([x + 0.2 for x in index], ['GaussianNB', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'AdaBoostClassifier',
              'BaggingClassifier', 'RandomForestClassifier' ])
    plt.legend()
    plt.show()

    # 时间
    plt.figure(figsize=(15, 8))
    plt.xlabel('classifier')
    plt.ylabel('cost(s)')
    plt.title('Time performance bar diagram')
    plt.bar(index, time_set, width=0.6, color="g", label = 'time')
    plt.xticks([x + 0.6 for x in index], ['GaussianNB', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'AdaBoostClassifier',
              'BaggingClassifier', 'RandomForestClassifier'])
    plt.legend()
    plt.show()