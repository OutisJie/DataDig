
# coding: utf-8

# In[1]:


import pandas as pd
import random as rd

from lshash.lshash import LSHash


# In[2]:


#读取文件
df = pd.read_csv('../reco_data/trade.csv')

#裁剪，并根据vipno、pluno分组，对amt求和
df1 = df.groupby([df.vipno, df.pluno])[['amt']].sum()
#这里将data转换为以vipno为行，pluno为列，便于之后处理
df2 = df1.unstack(0).fillna(0).round().transpose().loc['amt']
#将vipno转换为列，对不存在的值填0，数值四舍五入
data = df2.transpose()
#去掉行头和列头
del data.index.name
del data.transpose().index.name

data


# In[26]:


#将dataframe转换为纯数字矩阵
data_matrix = data.as_matrix()
col = data.shape[1]
row = data.shape[0]

for size in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
    hash_size = int(col * size)
    lsh = LSHash(hash_size, input_dim = row )
    
    #将矩阵导入，生成哈希表
    for  col_index in range(col):
        #将矩阵的每一列导入哈希表，将vipno的值作为extra_data
        lsh.index(data_matrix[:, col_index], extra_data = data.columns[col_index])
    else:
        #随机取列
        vipno_pos = 14
        for k in [1, 2, 3, 4, 5] :
            #得到hash_table的查询结果
            hash_table =  lsh.query(data_matrix[:, vipno_pos], num_results= k + 1, distance_func="euclidean")
            print("hash_size:" + str(hash_size) +
                  ", k:" + str(k) +
                  ", vipno:" +  str(data.columns[vipno_pos]) +
                  ", result:")
            result = []
            for res in hash_table:
                result.append(res[0][1])
            else:
                print(result[1:])

