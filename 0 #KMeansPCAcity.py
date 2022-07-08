# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 21:09:17 2021

@author: ASUS
"""

"""
object: cityData
operate: decompose PCA -> cluster KMeans
"""

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def loadData():
    print('CLUSTER'.center(30,'='))
    print('\n')
    print('LoadData'.center(30,'-'))
    print('\n')
    fp=input('filePath: ')
    print('\n')
    fr=open(fp,'r+')
    lines=fr.readlines()
    retData=[]
    retCityName=[]
    for line in lines:
        items=line.strip().split(',')
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1,len(items))])
    return retData,retCityName

def decompositionPCA(x):
    pca=PCA(n_components=2)
    reduced_x=pca.fit_transform(x)
    return reduced_x

def clusterKMeans(data):
    km=KMeans(n_clusters=3)
    labels=km.fit_predict(data)
    return labels

def printLabel(lab,datare):
    print('Labels'.center(30,'-'))
    print('\n')
    print(lab)
    print('\n')
    print('KMeans'.center(30,'-'))
    print('\n')
    n_clusters_=len(set(lab))
    print('Estimated number of clusters: %d'%n_clusters_)
    print('Silhouette Coefficient: %0.3f'%metrics.silhouette_score(datare,lab))
    print('\n')
    
def printCity(cityName,labels):
    clusterCity=[[],[],[]]
    for i in range(len(cityName)):
        clusterCity[labels[i]].append(cityName[i])
    for i in range(len(clusterCity)):
        print('Cluster ',i,':')
        print(clusterCity[i])
        print('\n')

def plotScatter(data_reduced,labels):
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    for i in range(len(data_reduced)):
        if labels[i] == 0:
            red_x.append(data_reduced[i][0])
            red_y.append(data_reduced[i][1])
        elif labels[i] == 1:
            blue_x.append(data_reduced[i][0])
            blue_y.append(data_reduced[i][1])
        else:
            green_x.append(data_reduced[i][0])
            green_y.append(data_reduced[i][1])
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='>')
    plt.show()
    print('END'.center(30,'='))

def main():
    data,cityName = loadData()
    data_reduced = decompositionPCA(data)
    labels = clusterKMeans(data_reduced)
    
    printLabel(labels,data_reduced)
    printCity(cityName,labels)
    plotScatter(data_reduced,labels)
    
main()