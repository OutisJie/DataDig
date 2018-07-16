import pandas as pd 
import sequentialFP

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

#提出pluno的数据
def pre_pluno(data):
    #丢弃多余的列
    data_pluno = data.drop(['sldat', 'uid', 'dptno', 'bndno'], axis = 1)
    
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
        s = []
        s.append([str(x) for x in row if x != 0.0])
        array_pluno.append(s)
    return array_pluno

#提出dptno的数据
def pre_dptno(data):
    #丢弃多余的列
    data_dptno = data.drop(['sldat', 'uid', 'pluno', 'bndno'], axis = 1)
    
    #合并订单
    data_dptno['value'] = data_dptno['dptno']
    data_dptno = data_dptno.pivot_table(data_dptno , index = ['vipno'], columns = 'dptno')
    
    #整理
    data_dptno = data_dptno.fillna(0).transpose().loc['value'].transpose()
    del data_dptno.index.name
    del data_dptno.transpose().index.name
    
    #将dataframe转为array
    array_dptno = []
    for row in data_dptno.as_matrix():
        s = []
        s.append([str(x) for x in row if x != 0.0])
        array_dptno.append(s)
    return array_dptno

#提出bndno的数据
def pre_bndno(data):
    #丢弃多余的列
    data_bndno = data.drop(['sldat', 'uid', 'pluno', 'dptno'], axis = 1)
    
    #合并订单
    data_bndno['value'] = data_bndno['bndno']
    data_bndno = data_bndno.pivot_table(data_bndno , index = ['vipno'], columns = 'bndno')
    
    #整理
    data_bndno = data_bndno.fillna(0).transpose().loc['value'].transpose()
    del data_bndno.index.name
    del data_bndno.transpose().index.name
    
    #将dataframe转为array
    array_bndno = []
    for row in data_bndno.as_matrix():
        s = []
        s.append([str(x) for x in row if x != 0.0])
        array_bndno.append(s)
    return array_bndno

def print_sets(patterns):
    for p in patterns:
        if len(p.sequence[0]) >= 2:
            print("pattern:{0}, support:{1}".format(p.sequence, p.support))

def fp_pluno(data):
    thresholds = [2, 4, 6, 8, 10]
    array_pluno = pre_pluno(data)
    for n in thresholds:
        freq_sets = sequentialFP.fpGrowth(array_pluno, n)
        print("for pluno, threshold: ", n)
        print_sets(freq_sets)
        
def fp_dptno(data):
    thresholds = [2, 4, 6, 8, 10]
    array_dptno = pre_dptno(data)
    for n in thresholds:
        freq_sets = sequentialFP.fpGrowth(array_dptno, n)
        print("for dptno, threshold: ", n)
        print_sets(freq_sets)
        
def fp_bndno(data):
    thresholds = [2, 4, 6, 8, 10]
    array_bndno = pre_bndno(data)
    for n in thresholds:
        freq_sets = sequentialFP.fpGrowth(array_bndno, n)
        print("for bndno, threshold: ", n)
        print_sets(freq_sets)