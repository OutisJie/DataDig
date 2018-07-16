import numpy as np
import pandas as pd

def pre_data(data):
    # 取出sldatime字段中的日期
    dates = []
    months = []
    for index, row in data.iterrows():
        date = row['sldatime'][5:10]
        month = row['sldatime'][5:7]
        dates.append(date)
        months.append(month)
 
    dates = pd.DataFrame(data=dates, columns=['date'])
    months = pd.DataFrame(data=months, columns=['month'])
    #print(dates)
    #print(months)
    data = pd.concat([data, dates, months], axis=1)

    return data

def type1_whole(data_set, feild, label):
    # 按照feild分组
    # 得到每个主体filed（u, b, c, i）的购买/被购买次数
    group_times = data_set[feild].value_counts()
    # 得到每个主体filed（u, b, c, i）的购买/被购买金额
    group_money = data_set.groupby([feild]).sum()
    # 得到每个主体filed（u, b, c, i）的购买/被购买天数
    group_days = data_set.groupby([feild, 'date']).size()

    counts = []
    moneys = []
    days = []

    for index, row in data_set.iterrows():
        index = row[feild]
        count = group_times.loc[index]
        counts.append(count)
        money = group_money.loc[index]['amt']
        moneys.append(money)
        day = group_days.loc[index].count()
        days.append(day)

    # 给data_set添加列,每个用户的购买次数
    counts = pd.DataFrame(data=counts, columns=[label + '_count_whole'])
    moneys = pd.DataFrame(data=moneys, columns=[label + '_money_whole'])
    days = pd.DataFrame(data=days, columns=[label + '_days_whole'])

    data_set = pd.concat([data_set, counts], axis=1)
    data_set = pd.concat([data_set, moneys], axis=1)
    data_set = pd.concat([data_set, days], axis=1)

    return data_set


def type1_month(data_set, feild, label):
    # 按照feild, month字段分组
    # 得到每个主体filed（u, b, c, i）每月的购买/被购买次数
    group_times = data_set.groupby([feild,'month']).size()
    # 得到每个主体filed（u, b, c, i）每月的购买/被购买金额
    group_money = data_set.groupby([feild, 'month']).sum()
    # 得到每个主体filed（u, b, c, i）每月的购买/被购买天数
    group_days = data_set.groupby([feild, 'month', 'date']).size()
    # print(group_times)
    # print(group_money)
    # print(group_days)
    counts = []
    moneys = []
    days = []

    for index, row in data_set.iterrows():
        index = row[feild]
        month = row['month']
        count = group_times.loc[index].loc[month]
        counts.append(count)
        money = group_money.loc[index].loc[month]['amt']
        moneys.append(money)
        day = group_days.loc[index].loc[month].count()
        days.append(day)

    # 给data_set添加列,每个用户的购买次数
    counts = pd.DataFrame(data=counts, columns=[label + '_count_month'])
    moneys = pd.DataFrame(data=moneys, columns=[label + '_money_month'])
    days = pd.DataFrame(data=days, columns=[label + '_days_month'])

    # print(counts)
    # print(moneys)
    # print(days)
    data_set = pd.concat([data_set, counts], axis=1)
    data_set = pd.concat([data_set, moneys], axis=1)
    data_set = pd.concat([data_set, days], axis=1)

    return data_set

def type1_whole_tw(data_set, feild1, feild2, label):
    # 按照feild1+feild2分组
    # 得到每个主体filed1+feild2（u, b, c, i）的购买/被购买次数
    group_times = data_set.groupby([feild1, feild2]).size()
    # 得到每个主体filed（u, b, c, i）的购买/被购买金额
    group_money = data_set.groupby([feild1, feild2]).sum()
    # 得到每个主体filed（u, b, c, i）的购买/被购买天数
    group_days = data_set.groupby([feild1, feild2, 'date']).size()

    counts = []
    moneys = []
    days = []

    for index, row in data_set.iterrows():
        index1 = row[feild1]
        index2 = row[feild2]
        count = group_times.loc[index1].loc[index2]
        counts.append(count)
        money = group_money.loc[index1].loc[index2]['amt']
        moneys.append(money)
        day = group_days.loc[index1].loc[index2].count()
        days.append(day)

    # 给data_set添加列
    counts = pd.DataFrame(data=counts, columns=[label + '_count_whole'])
    moneys = pd.DataFrame(data=moneys, columns=[label + '_money_whole'])
    days = pd.DataFrame(data=days, columns=[label + '_days_whole'])

    # print(counts)
    # print(moneys)
    # print(days)
    data_set = pd.concat([data_set, counts], axis=1)
    data_set = pd.concat([data_set, moneys], axis=1)
    data_set = pd.concat([data_set, days], axis=1)

    return data_set

def type1_month_tw(data_set, feild1, feild2, label):
    # 按照feild, month字段分组
    # 得到每个主体filed1+feild2（u, b, c, i）每月的购买/被购买次数
    group_times = data_set.groupby([feild1, feild2,'month']).size()
    # 得到每个主体filed1,2（u, b, c, i）每月的购买/被购买金额
    group_money = data_set.groupby([feild1, feild2, 'month']).sum()
    # 得到每个主体filed1,2（u, b, c, i）每月的购买/被购买天数
    group_days = data_set.groupby([feild1, feild2, 'month', 'date']).size()

    counts = []
    moneys = []
    days = []

    for index, row in data_set.iterrows():
        index1 = row[feild1]
        index2 = row[feild2]
        month = row['month']
        count = group_times.loc[index1].loc[index2].loc[month]
        counts.append(count)
        money = group_money.loc[index1].loc[index2].loc[month]['amt']
        moneys.append(money)
        day = group_days.loc[index1].loc[index2].loc[month].count()
        days.append(day)

    # 给data_set添加列,每个用户的购买次数
    counts = pd.DataFrame(data=counts, columns=[label + '_count_month'])
    moneys = pd.DataFrame(data=moneys, columns=[label + '_money_month'])
    days = pd.DataFrame(data=days, columns=[label + '_days_month'])

    data_set = pd.concat([data_set, counts], axis=1)
    data_set = pd.concat([data_set, moneys], axis=1)
    data_set = pd.concat([data_set, days], axis=1)

    return data_set

def type1_whole_pd(data_set, feild1, feild2, label):
    # 按照feild1,feild2分组, 统计不同的feild2出现的次数
    group_counts = data_set.groupby([feild1, feild2]).size()
    counts = []
    for index, row in data_set.iterrows():
        index = row[feild1]
        count = group_counts.loc[index].count()
        counts.append(count)
    counts = pd.DataFrame(data=counts, columns=[label + '_count_whole'])
    data_set = pd.concat([data_set, counts], axis=1)
    return data_set

def type1_month_pd(data_set, feild1, feild2, label):
    # 按照feild1,month,feild2分组
    group_counts = data_set.groupby([feild1, 'month', feild2]).size()
    counts = []
    for index, row in data_set.iterrows():
        index = row[feild1]
        month = row['month']
        counts.append(group_counts.loc[index].loc[month].count())
    counts = pd.DataFrame(data=counts, columns=[label + '_count_month'])
    data_set = pd.concat([data_set, counts], axis=1)
    return data_set

def type1(data):
    data_set = pre_data(data)
    # ---------count---------------
    # 1. 按user分组
    # 1.1 按whole period 统计
    data_set = type1_whole(data_set, 'vipno', 'user')
    # 1.2 按monthly period统计
    data_set = type1_month(data_set, 'vipno', 'user')
    # 2. 按brand分组
    # 2.1 按whole统计
    data_set = type1_whole(data_set, 'bndno', 'brand')
    # 2.2 按month统计
    data_set = type1_month(data_set, 'bndno', 'brand')
    # 3. 按categories分组
    data_set = type1_whole(data_set, 'dptno', 'cate')
    data_set = type1_month(data_set, 'dptno', 'cate')
    # 4. 按item分组
    data_set = type1_whole(data_set, 'pluno', 'item')
    data_set = type1_month(data_set, 'pluno', 'item')
    # 5. 按user+brand
    data_set = type1_whole_tw(data_set, 'vipno', 'bndno', 'ub')
    data_set = type1_month_tw(data_set, 'vipno', 'bndno', 'ub')
    # 6. 按user+cate
    data_set = type1_whole_tw(data_set, 'vipno', 'dptno', 'uc')
    data_set = type1_month_tw(data_set, 'vipno', 'dptno', 'uc')
    # 7. 按user+item
    data_set = type1_whole_tw(data_set, 'vipno', 'pluno', 'ui')
    data_set = type1_month_tw(data_set, 'vipno', 'pluno', 'ui')
    # 8. 按brand+cate
    data_set = type1_whole_tw(data_set, 'bndno', 'dptno', 'bc')
    data_set = type1_month_tw(data_set, 'bndno', 'dptno', 'bc')

    #---------product diversity----------
    # 1. 按user分组
    # 1.1. unique item
    data_set = type1_whole_pd(data_set, 'vipno', 'pluno', 'ui_unique')
    data_set = type1_month_pd(data_set, 'vipno', 'pluno', 'ui_unique')
    # 1.2. unique brand
    data_set = type1_whole_pd(data_set, 'vipno', 'bndno', 'ub_unique')
    data_set = type1_month_pd(data_set, 'vipno', 'bndno', 'ub_unique')
    # 1.3. unique cate
    data_set = type1_whole_pd(data_set, 'vipno', 'dptno', 'uc_unique')
    data_set = type1_month_pd(data_set, 'vipno', 'dptno', 'uc_unique')
    # 2. 按brand分组
    # unique item
    data_set = type1_whole_pd(data_set, 'bndno', 'pluno', 'bi_unique')
    data_set = type1_month_pd(data_set, 'bndno', 'pluno', 'bi_unique')
    # 3. 按cate分组
    # unique item
    data_set = type1_whole_pd(data_set, 'dptno', 'pluno', 'ci_unique')
    data_set = type1_month_pd(data_set, 'dptno', 'pluno', 'ci_unique')

    #--------penetration----------------
    # 1. 按brand分组
    data_set = type1_whole_pd(data_set, 'bndno', 'vipno', 'bu_people')
    data_set = type1_month_pd(data_set, 'bndno', 'vipno', 'bu_people')
    # 2. 按cate分组
    data_set = type1_whole_pd(data_set, 'dptno', 'vipno', 'cu_people')
    data_set = type1_month_pd(data_set, 'dptno', 'vipno', 'cu_people')
    # 3. 按item分组
    data_set = type1_whole_pd(data_set, 'pluno', 'vipno', 'iu_people')
    data_set = type1_month_pd(data_set, 'pluno', 'vipno', 'iu_people')

    #print(data_set)
    return data_set