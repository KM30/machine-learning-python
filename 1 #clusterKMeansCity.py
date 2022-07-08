import  numpy as np
from  sklearn import metrics
from sklearn.cluster import KMeans

def loadData(fpath):
    fr=open(fpath,'r+')
    lines=fr.readlines()
    retData=[]
    retCityName=[]

    for line in lines:
        items=line.strip().split(',')
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1,len(items))])

    return retData,retCityName

def main():
    print('CLUSTER'.center(30,'='))
    print('\n')
    print('LoadData'.center(30,'-'))
    print('\n')
    fp=input('filePath: ')
    print('\n')
    data,cityName=loadData(fp)
    km=KMeans(n_clusters=3)
    labels=km.fit_predict(data)
    print('Labels'.center(30,'-'))
    print('\n')
    print(labels)
    print('\n')
    expenses=np.sum(km.cluster_centers_,axis=1)
    clusterCity=[[],[],[]]

    for i in range(len(cityName)):
        clusterCity[labels[i]].append(cityName[i])

    print('KMeans'.center(30,'-'))
    print('\n')
    n_clusters_=len(set(labels))
    print('Estimated number of clusters: %d'%n_clusters_)
    print('Silhouette Coefficient: %0.3f'%metrics.silhouette_score(data,labels))
    print('\n')

    for i in range(len(clusterCity)):
        print('Cluster ',i,':')
        print('Expenses:%.2f' % expenses[i])
        print(clusterCity[i])
        print('\n')
        
    print('END'.center(30,'='))
    
main()
