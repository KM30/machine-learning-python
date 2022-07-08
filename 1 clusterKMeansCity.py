"""
-*- coding: utf-8 -*-

Created on Wed Dec 23 11:25:53 2020

@author: KM30

@codeLine:46

@CityData

"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics


# 1 getData API #

print('CLUSTER'.center(30,'='))
print('\n')
print('LoadData'.center(30,'-'))
fp=input('\n''filePath: ')
print('\n')

#lines列表一维,cityData的每一行为一个字符串str元素
lines=open(fp,'r+').readlines() 

retData=[]
retCityName=[]

for line in lines: 
    #line字符串类型,及cityData的一行
    items=line.strip().split(',') 
    #strip()删除字符串中的空白符,将line大字符串分割为更小的字符串,存入列表items
    retCityName.append(items[0]) 
    #items[i]取出i处的字符串元素,存入一维列表中
    retData.append([float(items[i]) for i in range(1,len(items))])
    #append每次添加一个一维列表,float()将字符串转换为整型


# 2 analyseData/processData #

km=KMeans(n_clusters=3)
#创建实例聚类中心个数,欧式距离
label=km.fit_predict(retData)
#data包含样本个数与特征个数，label是一维数组,其值与聚类中心个数有关,0,1,2.

print('Labels'.center(30,'-'))
print('\n')
print(label)
print('\n')

expenses=np.sum(km.cluster_centers_,axis=1)
#km.cluster_centers_是二维数组,axis=1&0,1按行 0按列
cityCluster=[[],[],[]]
for i in range(len(retCityName)):
    cityCluster[label[i]].append(retCityName[i])
    #label[i]存储了第i个样本对应的标签,样本的特征与名字是通过角标[i]建立的映射    


# 3 displayData/showData #

print('KMeans'.center(30,'-'))
print('\n')

n_clusters_=len(set(label))
print('Estimated number of clusters: %d'%n_clusters_)
print('Silhouette Coefficient: %0.3f'%metrics.silhouette_score(retData,label))
print('\n')

for i in range(len(cityCluster)):
    print('Cluster ',i,':')
    print("Accumulation: %.2f" % expenses[i])
    #格式化输出类型3:'%'可理解为数字映射符
    print(cityCluster[i]) 
    #打印输出类型为列表
    print('\n')
    
print('END'.center(30,'='))