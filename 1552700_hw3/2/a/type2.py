import pandas as pd
import numpy as np


def type2_process1(data):
    feilds = data.columns[25:]
    for feild in feilds:
        if feild.count('_month') > 0:
            grouped = data[feild].groupby(data['month'])

            # mean
            group_mean = grouped.mean()
            # std
            group_std = grouped.std()
            # max
            group_max = grouped.max()
            # median
            group_median = grouped.median()

            means = []
            stds = []
            maxs = []
            medians = []

            for index, row in data.iterrows():
                means.append(group_mean.loc[row['month']])
                stds.append(group_std.loc[row['month']])
                maxs.append(group_max.loc[row['month']])
                medians.append(group_median.loc[row['month']])
            # means = np.array(means).transpose()
            # print(means)
            means = pd.DataFrame(data=means, columns=[feild + '_mean'])
            stds = pd.DataFrame(data=stds, columns=[feild + '_std'])
            maxs = pd.DataFrame(data=maxs, columns=[feild + '_max'])
            medians = pd.DataFrame(data=medians, columns=[feild + '_median'])

            data = pd.concat([data, means], axis=1)     
            data = pd.concat([data, stds], axis=1) 
            data = pd.concat([data, maxs], axis=1)
            data = pd.concat([data, medians], axis=1)
            
    return data

def type2_process2_days(data_set, feild1, feild2, label):
    group_day = data_set.groupby([feild1, feild2, 'date']).size()
    group_day = pd.Series(data=1, index=group_day.index)

    new_group = []
    for index, row in data_set.drop_duplicates([feild1, feild2]).iterrows():
        new_group.append([row[feild1], row[feild2], group_day.loc[row[feild1]].loc[row[feild2]].count()])
    new_group = pd.DataFrame(data=new_group, columns=[feild1, feild2, label])
    new_group = new_group.groupby([new_group[feild1]])

    # mean
    group_mean = new_group.mean()
    # std
    group_std = new_group.std()
    # max
    group_max = new_group.max()
    # median
    group_median = new_group.median()

    means = []
    stds = []
    maxs = []
    medians = []

    for index, row in data_set.iterrows():
        means.append(group_mean.loc[row[feild1]][label])
        stds.append(group_std.loc[row[feild1]][label])
        maxs.append(group_max.loc[row[feild1]][label])
        medians.append(group_median.loc[row[feild1]][label])

    means = pd.DataFrame(data=means, columns=[label + '_mean'])
    stds = pd.DataFrame(data=stds, columns=[label + '_std'])
    maxs = pd.DataFrame(data=maxs, columns=[label + '_max'])
    medians = pd.DataFrame(data=medians, columns=[label + '_median'])

    data_set = pd.concat([data_set, means], axis=1)
    data_set = pd.concat([data_set, stds], axis=1)
    data_set = pd.concat([data_set, maxs], axis=1)
    data_set = pd.concat([data_set, medians], axis=1)

    return data_set

def type2_process2_times_or_money(data_set, feild1, feild2, label):
    group_label = data_set.groupby([feild1, feild2]).size()
    if label.count('money') > 0:
        group_label = data_set['amt'].groupby([data_set[feild1], data_set[feild2]]).sum()

    means = []
    stds = []
    maxs = []
    medians = []

    for index, row in data_set.iterrows():
        temp = group_label.loc[row[feild1]]
        means.append(temp.mean())
        stds.append(temp.std())
        maxs.append(temp.max())
        medians.append(temp.median())

    means = pd.DataFrame(data=means, columns=[label + '_mean'])
    stds = pd.DataFrame(data=stds, columns=[label + '_std'])
    maxs = pd.DataFrame(data=maxs, columns=[label + '_max'])
    medians = pd.DataFrame(data=medians, columns=[label + '_median'])

    data_set = pd.concat([data_set, means], axis=1)
    data_set = pd.concat([data_set, stds], axis=1)
    data_set = pd.concat([data_set, maxs], axis=1)
    data_set = pd.concat([data_set, medians], axis=1)

    return data_set

def type2(data):
    #------------month AGG------------
    data_set = type2_process1(data)
    #------------user AGG-------------
    # 1. 对brand分组
    # 1.1. 对于user, 发生购买的天数
    data_set = type2_process2_days(data_set, 'bndno', 'vipno', 'bu_days')
    # 1.2. 购买的次数
    data_set = type2_process2_times_or_money(data_set, 'bndno', 'vipno', 'bu_times')
    # 1.3. 购买的金额
    data_set = type2_process2_times_or_money(data_set, 'bndno', 'vipno', 'bu_money')
    # 2. 对cate分组
    data_set = type2_process2_days(data_set, 'dptno', 'vipno', 'cu_days')
    data_set = type2_process2_times_or_money(data_set, 'dptno', 'vipno', 'cu_times')
    data_set = type2_process2_times_or_money(data_set, 'dptno', 'vipno', 'cu_money')
    # 3. 对item分组
    data_set = type2_process2_days(data_set, 'pluno', 'vipno', 'iu_days')
    data_set = type2_process2_times_or_money(data_set, 'pluno', 'vipno', 'iu_times')
    data_set = type2_process2_times_or_money(data_set, 'pluno', 'vipno', 'iu_money')
    #-----------brand/category/item AGG---------
    # 1.对user分组，统计brand
    data_set = type2_process2_days(data_set, 'vipno', 'bndno', 'ub_days')
    data_set = type2_process2_times_or_money(data_set, 'vipno', 'bndno', 'ub_times')
    data_set = type2_process2_times_or_money(data_set, 'vipno', 'bndno', 'ub_money')
    # 2.对user分组，统计cate
    data_set = type2_process2_days(data_set, 'vipno', 'dptno', 'uc_days')
    data_set = type2_process2_times_or_money(data_set, 'vipno', 'dptno', 'uc_times')
    data_set = type2_process2_times_or_money(data_set, 'vipno', 'dptno', 'uc_money')
    # 1.对user分组，统计item
    data_set = type2_process2_days(data_set, 'vipno', 'pluno', 'ui_days')
    data_set = type2_process2_times_or_money(data_set, 'vipno', 'pluno', 'ui_times')
    data_set = type2_process2_times_or_money(data_set, 'vipno', 'pluno', 'ui_money')
    
    return data_set
