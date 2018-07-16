
# coding: utf-8

# In[8]:


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



#提出dptno的数据
def pre_dptno(data):
    #丢弃多余的列
    data_dptno = data.drop(['sldat', 'vipno', 'pluno', 'bndno'], axis = 1)
    
    #合并订单
    data_dptno['value'] = data_dptno['dptno']
    data_dptno = data_dptno.pivot_table(data_dptno , index = ['uid'], columns = 'dptno')
    
    #整理
    data_dptno = data_dptno.fillna(0).transpose().loc['value'].transpose()
    del data_dptno.index.name
    del data_dptno.transpose().index.name
    
    #将dataframe转为array
    array_dptno = []
    for row in data_dptno.as_matrix():
        array_dptno.append([x for x in row if x != 0.0])
    return array_dptno


# In[6]:



def fp_dptno(data):
    thresholds = [2, 4, 8, 16, 32, 64]
    array_dptno = pre_dptno(data)
    for n in thresholds:
        freq_sets = fpGrowth.fpGrowth(array_dptno, n)
        print("for dptno, threshold: ", n)
        for k in freq_sets:
            if len(k) >= 2:
                print (k, freq_sets[k])


# In[9]:


if __name__ == "__main__":
    old_df = pd.read_csv('../trade.csv', usecols=['uid', 'vipno', 'sldat', 'pluno', 'dptno', 'bndno'])
    #旧数据
    old_data = pre_data(old_df)
    start = time.clock()
    fp_dptno(old_data)
    time_old = time.clock() - start
    #新数据
    new_df = pd.read_csv('../trade_new.csv', usecols=['uid', 'vipno', 'sldatime', 'pluno', 'dptno', 'bndno'])
    new_df.rename(columns={ new_df.columns[1]: "sldat" }, inplace=True)    
    new_data = pre_data(new_df)
    start = time.clock()
    fp_dptno(new_data)
    time_new = time.clock() - start
        
    print("fp_dptno,for old data: ", time_old)
    print("fp_dptno,for new data: ", time_new)

