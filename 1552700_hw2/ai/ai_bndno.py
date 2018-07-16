
# coding: utf-8

# In[9]:


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



#提出bndno的数据
def pre_bndno(data):
    #丢弃多余的列
    data_bndno = data.drop(['sldat', 'vipno', 'pluno', 'dptno'], axis = 1)
    
    #合并订单
    data_bndno['value'] = data_bndno['bndno']
    data_bndno = data_bndno.pivot_table(data_bndno , index = ['uid'], columns = 'bndno')
    
    #整理
    data_bndno = data_bndno.fillna(0).transpose().loc['value'].transpose()
    del data_bndno.index.name
    del data_bndno.transpose().index.name
    
    #将dataframe转为array
    array_bndno = []
    for row in data_bndno.as_matrix():
        array_bndno.append([x for x in row if x != 0.0])
    return array_bndno


# In[7]:



def fp_bndno(data):
    thresholds = [2, 4, 8, 16, 32, 64]
    array_bndno = pre_bndno(data)
    for n in thresholds:
        freq_sets = fpGrowth.fpGrowth(array_bndno, n)
        print("for bndno, threshold: ", n)
        for k in freq_sets:
            if len(k) >= 2:
                print (k, freq_sets[k])


# In[10]:


if __name__ == "__main__":
    old_df = pd.read_csv('../trade.csv', usecols=['uid', 'vipno', 'sldat', 'pluno', 'dptno', 'bndno'])
    #旧数据
    old_data = pre_data(old_df)
    start = time.clock()
    fp_bndno(old_data)
    time_old = time.clock() - start;
    
    #新数据
    new_df = pd.read_csv('../trade_new.csv', usecols=['uid', 'vipno', 'sldatime', 'pluno', 'dptno', 'bndno'])
    new_df.rename(columns={ new_df.columns[1]: "sldat" }, inplace=True)    
    new_data = pre_data(new_df)
    start = time.clock()
    fp_bndno(new_data)
    time_new = time.clock() - start;
    
    print("fp_bndno,for old data: ", time_old)
    print("fp_bndno,for new data: ", time_new)

