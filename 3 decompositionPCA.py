import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
#以字典类型加载鸢尾花数据
y = data.target
#y表示数据集的标签共三类,shape(150,1) array of int32
X = data.data
#x表示数据集的属性值,shape(150,4) array of float64

pca = PCA(n_components=2)
#创建PCA算法实例,设置主成分数目2,即降维后单个样本维度为2
reduced_X = pca.fit_transform(X)
#加载数据进行训练,将样本维度由4降为2,shape(150,2) array of float64

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_X)):
    #按标签y统计数据
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
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