# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:55:32 2020

@author: ASUS
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

def loadIris():
    data=load_iris()
    y=data.target
    x=data.data
    return y,x

def decompositionPCA(x2):
    pca=PCA(n_components=2)
    reduced_x=pca.fit_transform(x2)
    return reduced_x

def plotScatter(reduced_x2,y2):
    rx,ry=[],[]
    gx,gy=[],[]
    bx,by=[],[]
    for i in range(len(reduced_x2)):
        if y2[i]==0:
            rx.append(reduced_x2[i][0])
            ry.append(reduced_x2[i][1])
        elif y2[i]==1:
            gx.append(reduced_x2[i][0])
            gy.append(reduced_x2[i][1])
        else:
            bx.append(reduced_x2[i][0])
            by.append(reduced_x2[i][1])
    plt.scatter(rx,ry,c='r',marker='x')
    plt.scatter(gx,gy,c='g',marker='<')
    plt.scatter(bx,by,c='b',marker='D')
    plt.show()
    
def main():
    y1,x1=loadIris()
    reduced_x1=decompositionPCA(x1)
    plotScatter(reduced_x1,y1)
    
main()