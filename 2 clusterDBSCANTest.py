import numpy as np
from  sklearn import metrics
import sklearn.cluster as skc
import matplotlib.pyplot as plt


# 1 getData API #

print('CLUSTER'.center(30,'='))
print('\n')
print('LoadData'.center(30,'-'))
print('\n')
fpath=input('filePath: ')
print('\n')

f=open(fpath,encoding='utf-8')

mac2id=dict() 
#创建字典
onlinetimes=[] 
#创建列表

for line in f:
    mac=line.split(',')[2] 
    #提取mac地址字符串类型
    onlinetime=int(line.split(',')[6]) 
    #提取上网时长字符串转换为整型eval()
    starttime=int(line.split(',')[4].split(' ')[1].split(':')[0]) 
    #提取上网开始时间
    if mac not in mac2id:
        mac2id[mac]=len(onlinetimes) 
        #每个关键字mac对应值为0,1,2,.序列数
        onlinetimes.append((starttime,onlinetime)) 
        #列表里的元素为元组类型
    else:
        onlinetimes[mac2id[mac]]=(starttime,onlinetime)
      
#real_X=np.array(onlinetimes).reshape((-1,2)) 
real_X=np.array(onlinetimes)
#将列表转换为数组,已知矩阵列数,每行2个数分完为止
X=real_X[:,0:1] 
#数组切片和索引,[:,0]索引值结果为一维数组,[:,0:1]切片列表结果为二维数组


# 2 analyseData/processData #

db=skc.DBSCAN(eps=0.01,min_samples=20).fit(X) 
#Expected 2D array,创建DBSCAN算法实例,并进行训练,曼哈顿距离
labels = db.labels_
#labels是一维列表,其值包括-1(表示噪音点),1(表示聚类1),2,...

print('Labels'.center(30,'-'))
print('\n')
print(labels)
print('\n')

print('DBSCAN'.center(30,'-'))
print('\n')

raito=len(labels[labels[:] == -1]) / len(labels)
#labels[:] == -1,返回一维array labels里为-1的全部元素的位置对应值为True 如下：
#array([ True,  True,  True,  True,  True,  True, False, False, False,
#       False, False, False, False, False, False, False, False, False])
#len(labels[labels[:] == -1]),计算labels里为-1的全部元素的个数
#注意该方法与set()函数的对比
print('Noise raito:',format(raito, '.2%'))
#格式化输出类型1:('{*}'.format(*)) & 类型2:('*',format(*,'*')),%是转换为百分数

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#set()函数创建一个无序不重复元素的集合,删除重复的元素,计算簇的个数


# 3 displayData/showData #

print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels))
#聚类效果评价指标,silhoustte coefficient轮廓系数,metrics指标

print('\n')

for i in range(n_clusters_):
    print('Cluster ',i,':')
    print(list(X[labels == i].flatten()))
    #X[labels==i]提取簇i的样本元素,flatten()数组降维返回一个折叠成一维的数组

print('\n')
print('END'.center(30,'='))

plt.hist(X,24)
#hist()适合有重复及相同元素,统计比例时用