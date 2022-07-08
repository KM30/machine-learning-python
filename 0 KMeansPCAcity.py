# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:25:53 2020

@author: ASUS
"""

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


# 1 getData API #
print('CLUSTER'.center(30,'='))
print('\n')
print('LoadData'.center(30,'-'))
print('\n')
fp=input('filePath: ')
print('\n')

rf=open(fp,'r+')
lines=rf.readlines() 
#以每行为字符串str元素存入一维列表

retData=[]
retCityName=[]

for line in lines: 
    #line为字符串类型
    items=line.strip().split(',') 
    #strip()删除字符串中的空白符
    retCityName.append(items[0]) 
    #一维列表
    retData.append([float(items[i]) for i in range(1,len(items))])
    #append每次添加一个一维列表



# 2 analyseData/processData #
pca = PCA(n_components=2)
#创建PCA算法实例,设置主成分数目2,即降维后单个样本维度为2
reduced_X = pca.fit_transform(retData)
#加载数据进行训练,将样本维度由高降为2,shape(150,2) array of float64

km=KMeans(n_clusters=3) 
#创建实例聚类中心个数,欧式距离
label=km.fit_predict(reduced_X)
#data包含样本个数与维度信息，label是一维列表,其值与聚类中心个数有关,0,1,2.

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

# plt.hist(expenses,3)
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_X)):
    #按标签y统计数据
    if label[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif label[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
        
plt.scatter(red_x, red_y, c='r', marker='x')
#scatter散点图,x y同维度的数组样本数据
plt.scatter(blue_x, blue_y, c='b', marker='D')
#参数c为颜色或颜色序列
plt.scatter(green_x, green_y, c='g', marker='>')
#参数marker为样本点显示样式
plt.show()
