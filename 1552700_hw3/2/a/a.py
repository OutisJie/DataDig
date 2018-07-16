import numpy as np
import pandas as pd
import type1
import type2
import type3
import type4


if __name__ == '__main__':
    data = pd.read_csv('../trade_new.csv')
    data = data.drop(['Unnamed: 0'], axis=1).fillna(-1)

    # 64
    data = type1.type1(data)
    print(data.columns.values)
    # 200
    data = type2.type2(data)
    print(data.columns.values)
    # 32
    data = type3.type3(data)
    print(data.columns.values)
    # 64+12
    data = type4.type4(data)
    print(data.columns.values)
    
    data.to_csv('output.csv')