import numpy as np
import sys
import cv2
import matplotlib.pyplot as pyplot
import os
import time
np.set_printoptions(suppress=True)
inputName = sys.argv[1]
outputName = sys.argv[2]
f = open(inputName)
line = f.readline()
data_list = []
while line:
    num = list(map(float,line.split()))
    data_list.append(num)
    line = f.readline()
f.close()
#data_array1 = np.array(data_list,dtype=np.int)
data_array = np.array(num,np.uint8)#这个unit8很重要
step = 19
array = [data_array[i:i+step] for i in range(0,len(data_array),step)]
arraytoMat = np.mat(array)
#高斯模糊
#高斯矩阵尺寸和标准差
result = []
kernel_size = [(5, 5),(3,3),(7,7)]
sigma = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2]
for i in kernel_size:
    for j in sigma:
        localTime = time.strftime("%Y%m%d%H%M%S",time.localtime())
        filename = outputName+'//'+localTime+inputName

        np.savetxt(filename,(cv2.GaussianBlur(arraytoMat, i, j)),fmt = ['%s']*19)
        print("file"+" "+str(i)+":"+str(localTime)+".txt")
        time.sleep(1)
print(arraytoMat)
print('------------------------------------------------------------------------------------------------')
print(result)
'''cv2.imshow("result",result)
cv2.imshow("source",arraytoMat)
cv2.waitKey(0)'''
