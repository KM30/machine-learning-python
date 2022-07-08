from numpy.random import RandomState
#从numpy中加载RandomState用于创建随机种子
import matplotlib.pyplot as plt
#加载matplotlib用于数据可视化
from sklearn.datasets import fetch_olivetti_faces
#从sklearn数据集中加载原始数据olivettiFace
from sklearn import decomposition
#从sklearn中引入降维算法库如PCA,NMF

n_row, n_col = 2, 3
#设置图像展示时的排列情况, 2行3列
n_components = n_row * n_col
#设置提取的特征的数目,为6
image_shape = (64, 64)
#设置人脸数据图片的大小, *.reshape(image_shape)

###############################################################################

## Load faces data ##
dataset = fetch_olivetti_faces(shuffle=True, random_state=RandomState(0))
#以字典类型加载原始数据,并打乱顺序
faces = dataset.data
#提取标签 'data' 对应的数据array(400,4096),4096=64*64

###############################################################################

def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    # plt.figure 设置图形显示窗口特征,图片大小
    # figsize(float, float); Width,height in inches
    plt.suptitle(title, size=16)
    #绘制总标题,设置标题和字号大小
    for i,comp in enumerate(images):
        #enumerate()是python的内置函数
        #enumerate将可遍历的对象[列表,字符串]组成一个索引序列
        #利用它可以同时获得 索引 和 值
        plt.subplot(n_row, n_col, i + 1)
        #划分子绘图区域,前两个参数表示先行后列
        #第三个参数表示第i+1个子区域[标号从1开始]
        vmax = max(comp.max(), -comp.min())
        #
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest', vmin=-vmax, vmax=vmax)
        #对图形归一化; 以灰度图形式显示;
        #使用vmin和vmax参数或norm参数来控制[如果要非线性缩放]
        plt.xticks(())
        #去除子图的x坐标轴标签
        plt.yticks(())
        #去除子图的y坐标轴标签
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.94, 0.04, 0.)
    #对子图的位置及间隔进行调整
plot_gallery("First centered Olivetti faces", faces[:n_components])

###############################################################################

estimators = [
    ('Non-negative components - NMF',
         decomposition.NMF(n_components=6, init='nndsvda', tol=5e-3)),
    
    ('Eigenfaces - PCA using randomized SVD',
         decomposition.PCA(n_components=6,whiten=True))
]        #whiten=True,若一矩阵mxn,压缩方向为减小m

###############################################################################

for name, estimator in estimators:
    print("Extracting the top %d %s..." % (n_components, name))
    print(faces.shape)
    estimator.fit(faces)
    components_ = estimator.components_
    plot_gallery(name, components_[:n_components])

plt.show()