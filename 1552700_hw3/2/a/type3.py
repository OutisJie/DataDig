import pandas as pd

def get_last_month(current_month):
    months = ['02', '03', '04', '05', '06', '07']
    # 如果是第一个月，那就取最后一个月填补
    if months[0] == current_month:
        return months[len(months) - 1]
    for i in range(0, len(months)):
        if months[i] == current_month:
            return months[i-1]


def type3_type1_last_month(data_set, feild, label):
    # 按照feild, month字段分组
    # 得到每个主体filed（u, b, c, i）每月的购买/被购买次数
    group_times = data_set.groupby([feild,'month']).size()
    # 得到每个主体filed（u, b, c, i）每月的购买/被购买金额
    group_money = data_set.groupby([feild, 'month']).sum()
    # 得到每个主体filed（u, b, c, i）每月的购买/被购买天数
    group_days = data_set.groupby([feild, 'month', 'date']).size()

    counts = []
    moneys = []
    days = []

    for index, row in data_set.iterrows():
        index = row[feild]
        month = row['month']
        # 当前所在月month的上一个月last_month
        last_month = get_last_month(month)
        if last_month in group_times.loc[index].index:
            count = group_times.loc[index].loc[last_month]
            money = group_money.loc[index].loc[last_month]['amt']
            day = group_days.loc[index].loc[last_month].count()
        else:
            count = 0
            money = 0
            day = 0
        counts.append(count)
        moneys.append(money)
        days.append(day)

    # 给data_set添加列,每个用户的购买次数
    counts = pd.DataFrame(data=counts, columns=[label + '_count_last_month'])
    moneys = pd.DataFrame(data=moneys, columns=[label + '_money_last_month'])
    days = pd.DataFrame(data=days, columns=[label + '_days_last_month'])

    data_set = pd.concat([data_set, counts], axis=1)
    data_set = pd.concat([data_set, moneys], axis=1)
    data_set = pd.concat([data_set, days], axis=1)

    return data_set

def type3_type1_last_month_tw(data_set, feild1, feild2, label):
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
        last_month = get_last_month(month)
        if last_month in group_times.loc[index1].loc[index2].index:
            count = group_times.loc[index1].loc[index2].loc[last_month]
            money = group_money.loc[index1].loc[index2].loc[last_month]['amt']
            day = group_days.loc[index1].loc[index2].loc[last_month].count()
        else:
            count = 0
            money = 0
            day = 0
        counts.append(count)
        moneys.append(money) 
        days.append(day)

    # 给data_set添加列,每个用户的购买次数
    counts = pd.DataFrame(data=counts, columns=[label + '_count_last_month'])
    moneys = pd.DataFrame(data=moneys, columns=[label + '_money_last_month'])
    days = pd.DataFrame(data=days, columns=[label + '_days_last_month'])

    data_set = pd.concat([data_set, counts], axis=1)
    data_set = pd.concat([data_set, moneys], axis=1)
    data_set = pd.concat([data_set, days], axis=1)

    return data_set


def type3_type1_last_month_pd(data_set, feild1, feild2, label):
    # 按照feild1,month,feild2分组
    group_counts = data_set.groupby([feild1, 'month', feild2]).size()
    counts = []
    for index, row in data_set.iterrows():
        index = row[feild1]
        month = row['month']
        last_month = get_last_month(month)
        if last_month in group_counts.loc[index]:
            counts.append(group_counts.loc[index].loc[last_month].count())
        else:
            counts.append(0)

    counts = pd.DataFrame(data=counts, columns=[label + '_count_last_month'])
    data_set = pd.concat([data_set, counts], axis=1)
    return data_set

def type3(data):
    # 统计上一个月的数据
    # type1
    # --------count------------
    data_set = type3_type1_last_month(data, 'vipno', 'user')
    data_set = type3_type1_last_month(data_set, 'bndno', 'brand')
    data_set = type3_type1_last_month(data_set, 'dptno', 'cate')
    data_set = type3_type1_last_month(data_set, 'pluno', 'item')
    data_set = type3_type1_last_month_tw(data_set, 'vipno', 'bndno', 'ub')
    data_set = type3_type1_last_month_tw(data_set, 'vipno', 'dptno', 'uc')
    data_set = type3_type1_last_month_tw(data_set, 'vipno', 'pluno', 'ui')
    data_set = type3_type1_last_month_tw(data_set, 'bndno', 'dptno', 'bc')
    #---------product diversity------------
    data_set = type3_type1_last_month_pd(data_set, 'vipno', 'pluno', 'ui_unique')
    data_set = type3_type1_last_month_pd(data_set, 'vipno', 'bndno', 'ub_unique')
    data_set = type3_type1_last_month_pd(data_set, 'vipno', 'dptno', 'uc_unique')
    data_set = type3_type1_last_month_pd(data_set, 'bndno', 'pluno', 'bi_unique')
    data_set = type3_type1_last_month_pd(data_set, 'dptno', 'pluno', 'ci_unique')
    #----------penetration----------------
    data_set = type3_type1_last_month_pd(data_set, 'bndno', 'vipno', 'bu_unique')
    data_set = type3_type1_last_month_pd(data_set, 'dptno', 'vipno', 'cu_unique')
    data_set = type3_type1_last_month_pd(data_set, 'pluno', 'vipno', 'iu_unique')

    return data_set