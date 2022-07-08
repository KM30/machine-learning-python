# step 1 # 导入功能包 #
import numpy as np
import PIL.Image as image                 #加载pillow包，用于加载创建图片
from sklearn.cluster import KMeans        #加载聚类算法，进行图像分割


# step 2 # 加载图片并预处理 #
def loadData(filePath):
    f = open(filePath,'rb')               #以二进制形式打开文件
    data = []
    img = image.open(f)                   #以列表形式返回图像像素值
    m,n = img.size                        #获取图片大小
    for i in range(m):                    #将每个像素点RGB值处理到0-1范围内
        for j in range(n):
            x,y,z = img.getpixel((i,j))               #获取像素点RGB值
            data.append([x/256.0,y/256.0,z/256.0])    #存入data列表里
    f.close()
    return np.mat(data),m,n               #以矩阵形式返回data，以及图片大小

print('Image segmentation based on clustering'.center(60,'='),'\n')
fp=input('Please input the direction of the image: ')
print('\n')
imgData,row,col = loadData(fp)


# step 3 # 对像素点进行聚类，实现图像分割 #
label = KMeans(n_clusters=4).fit_predict(imgData)    #聚类获得每个像素所属类别
label = label.reshape([row,col])                    
pic_new = image.new("L", (row, col))                 #创建新的灰度图存储聚类结果
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i,j), int(256/(label[i][j]+1)))  #确保具有区分度即可
      
 
# step 4 # 输出结果 #
pic_new.save("result-bull.jpg", "JPEG")            #以JPEG格式保存图像
print('File is stored in: D:\MyData\Code\1 ML\1 Python\1 project\0 codepy\4 MachineLearning\n')
print('END'.center(60,'='))