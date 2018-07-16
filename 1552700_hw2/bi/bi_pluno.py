
# coding: utf-8

# In[6]:


import pandas as pd
import time
import bi_model


# In[7]:


if __name__ == "__main__":
    old_df = pd.read_csv('../trade.csv', usecols = ['uid', 'vipno', 'sldat', 'pluno', 'dptno', 'bndno'])
    old_data = bi_model.pre_data(old_df)
    start = time.clock()
    bi_model.fp_pluno(old_data)
    time_old = time.clock() - start
    
    new_df = pd.read_csv('../trade_new.csv', usecols=['uid', 'vipno', 'sldatime', 'pluno', 'dptno', 'bndno'])
    new_df.rename(columns={ new_df.columns[1]: "sldat" }, inplace=True)  
    new_data = bi_model.pre_data(new_df)
    start = time.clock()
    bi_model.fp_pluno(new_data)
    time_new = time.clock() - start
    
    print("for pluno, old data:", time_old)
    print("for pluno, new data:", time_new)

