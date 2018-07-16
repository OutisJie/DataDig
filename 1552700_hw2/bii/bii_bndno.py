
# coding: utf-8

# In[1]:


import pandas as pd
import time
import bii_model


# In[2]:


if __name__ == "__main__":
    old_df = pd.read_csv('../trade.csv', usecols = ['uid', 'vipno', 'sldat', 'pluno', 'dptno', 'bndno'])
    old_data = bii_model.pre_data(old_df)
    start = time.clock()
    bii_model.fp_bndno(old_data)
    time_old = time.clock() - start
    new_df = pd.read_csv('../trade_new.csv', usecols=['uid', 'vipno', 'sldatime', 'pluno', 'dptno', 'bndno'])
    new_df.rename(columns={ new_df.columns[1]: "sldat" }, inplace=True)  
    new_data = bii_model.pre_data(new_df)
    start = time.clock()
    bii_model.fp_bndno(new_data)
    time_new = time.clock() - start
    
    print("for bndno, old data:", time_old)
    print("for bndno, new data:", time_new)

