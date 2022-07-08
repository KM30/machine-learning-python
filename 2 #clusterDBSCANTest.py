"""
-*-  coding: utf-8  -*-

Created on Mon Nov 30 14:11:50 2020

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import DBSCAN


def loadData(fp):
    mac2id=dict()
    onlinetimes=[]
    
    lines=open(fp,encoding='utf-8')
    for line in lines:
        mac=line.split(',')[2]
        onlinetime=int(line.split(',')[6])
        starttime=int(line.split(',')[4].split(' ')[1].split(':')[0])
        if mac not in mac2id:
            mac2id[mac]=len(onlinetimes)
            onlinetimes.append((starttime,onlinetime))
        else:
            onlinetimes[mac2id[mac]]=(starttime,onlinetime)
            
    olts=np.array(onlinetimes).reshape((-1,2))
    sts=olts[:,0:1]
    return sts  

def cDBSCAN(x):
    db=DBSCAN(eps=0.01,min_samples=20).fit(x)
    labels=db.labels_
    print(labels)
    print('\n')
    raito=len(labels[labels[:]==-1])/len(labels)
    print('DBSCAN'.center(30,'-'))
    print('\n')
    print('Noise raito: ',format(raito,'.2%'))
    n_clusters_=len(set(labels))-(1 if -1 in labels else 0)
    print('Estimated number of clusters: %d'%n_clusters_)
    print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(x, labels))
    print('\n')
    for i in range(n_clusters_):
        print('Cluster ',i,':')
        print(list(x[labels == i].flatten()))
        
def cPrint(x):
    plt.hist(x,24)
    
def main():
    print('CLUSTER'.center(30,'='))
    print('\n')
    print('LoadData'.center(30,'-'))
    print('\n')
    fpath=input('filePath: ')
    print('\n')
    st=loadData(fpath)
    print('Labels'.center(30,'-'))
    print('\n')
    cDBSCAN(st)
    print('\n')
    print('END'.center(30,'='))
    cPrint(st)
    
main()