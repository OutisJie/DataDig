
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
import fpGrowth 


# In[2]:


#预处理数据
def pre_data(df):
    df['timestamp'] = pd.to_datetime(df['sldat'])
    
    #分别按vipno、sldat排序
    data = df.sort_values(by=['vipno', 'sldat'])
    
    #分组
    data['rank'] = data['timestamp'].groupby(data['vipno']).rank(ascending=0, method='first')
    
    #取每个vipno的前60%的数据
    data = data.groupby(['vipno'], as_index = False).apply(lambda x: x[x['rank'] <=  round(0.6 * x['rank'].max())])
    #整理
    data = data.drop(['timestamp', 'rank'], axis = 1).reset_index(drop=True)
    return data


# In[3]:


#提出pluno的数据
def pre_pluno(data):
    #丢弃多余的列
    data_pluno = data.drop(['sldat', 'dptno', 'bndno'], axis = 1)
    
    #合并订单
    data_pluno['value'] = data_pluno['pluno']
    data_pluno = data_pluno.pivot_table(data_pluno , index = ['vipno'], columns = 'pluno')
    
    #整理
    data_pluno = data_pluno.fillna(0).transpose().loc['value'].transpose()
    del data_pluno.index.name
    del data_pluno.transpose().index.name
    
    #将dataframe转为array
    array_pluno = []
    for row in data_pluno.as_matrix():
        array_pluno.append([x for x in row if x != 0.0])
    return array_pluno


# In[6]:


def fp_pluno(data):
    thresholds = [2, 4, 6, 8, 10]
    array_pluno = pre_pluno(data)
    for n in thresholds:
        freq_sets = fpGrowth.fpGrowth(array_pluno, n)
        print("for pluno, threshold: ", n)
        for k in freq_sets:
            if len(k) >= 2:
                print(k, freq_sets[k])


# In[5]:


if __name__ == "__main__":
    old_df = pd.read_csv('../trade.csv', usecols=['vipno', 'sldat', 'pluno', 'dptno', 'bndno'])
    #旧数据
    old_data = pre_data(old_df)
    start = time.clock()
    fp_pluno(old_data)
    time_old = time.clock() - start
    
    #新数据
    new_df = pd.read_csv('../trade_new.csv', usecols=['vipno', 'sldatime', 'pluno', 'dptno', 'bndno'])
    new_df.rename(columns={ new_df.columns[0]: "sldat" }, inplace=True)    
    new_data = pre_data(new_df)
    start = time.clock()
    fp_pluno(new_data)
    time_new = time.clock() - start
    
    print("for pluno, old_data:" , time_old)
    print("for pluno, new_data:" , time_new)

