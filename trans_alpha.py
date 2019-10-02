import numpy as np
import os 
import re
import time
#from mydtw import compute_distances_no_loops
import matplotlib.pyplot as plt

#无循环(伪)dtw
def compute_distances_no_loops(A, B, trace):
    #A 是m*k矩阵，B是n*k矩阵。求m*n距离矩阵
    m = np.shape(A)[0]
    n = np.shape(B)[0]
    #dists = np.zeros((m,n))                                         
    M = np.dot(A, B.T)
    H1 = np.square(A).sum(axis = 1) #H1是1*m数组，也可看成1*m行向量
    H2 = np.square(B).sum(axis = 1) #H2是1*n维行向量
    D = np.sqrt(-2*M+H1+np.matrix(H2).T) #要将H2展开到D上的每一列，需要转置成列向量。
    D0 = np.zeros((m+1, n+1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    C = D.copy()
    D2 = np.zeros((np.shape(D)[0],1))
    D2[0:] = np.inf
    D3 = np.zeros((np.shape(D)[1]+1,1))
    D3[1:] = np.inf
    D4 = np.c_[D2,D]
    D5 = np.r_[D3.T,D4]
    D = D5[1:, 1:]
    for i in range(m):
        for j in range(n):
            D[i, j] += min(D5[i, j], D5[i, j+1], D5[i+1, j])
    if(trace == True):
        path = _traceback(D5)
        path_list = findpath(path,C)
        return D[-1, -1],path_list#,D,path
    else:
        return D[-1,-1]

#回溯找路径
def _traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    #path_list = []
    #path_list.append(C[i,j])
    while ((i > 0) or (j > 0)):
        tb = np.argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1 
        else: # (tb == 2):
            j -= 1
        #path_list.append(C[i,j])直接输出路径元素，出了点问题
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)
#输出矩阵路径元素
def findpath(path,M):
    len = np.shape(path)[1]
    path_list = []
    for count in range(0,len):
        i = path[0][count]
        j = path[1][count]
        path_list.append(M[i,j])
    return path_list

def second_min(seq):
    sortlist = sorted(seq)
    return sortlist[1]
#输出分粒匹配分数
def fenli(list):
    t = np.mean(list)
    match_num = sum(i<=t for i in list)
    match_Sum = sum([i for i in list if i<t])
    num = len(list)
    score = match_num/num
    #print(t,match_num,num)
    return score,match_Sum



#l2_norm = lambda x, y: (x - y) ** 2#lamda表达式，欧氏距离
seqs = []#创建空列表
mydict = {}#创建空字典
x = input()
xlist = x.split(" ")
xlist = [int(xlist[i]) for i in range(len(xlist))]
xlist = np.array(xlist)
xlist = [xlist[i:i+19] for i in range(0,len(xlist),19)]
xlist = np.mat(xlist)
#print(np.shape(xlist))
#test = list((np.loadtxt('/home/xuyt/文档/手语识别/777/爱/20190520171319_爱.txt')).flatten())
'''rootdir = '/home/xuyt/文档/手语识别/处理后'  #主文件目录
dirlist =os.listdir(rootdir) #列出文件夹下所有的目录与文件
for i in range(0,len(dirlist)):
    path = os.path.join(rootdir, dirlist[i])
    array = [np.loadtxt(path)[i:i+19] for i in range(0,len(np.loadtxt(path)),19)]
    #array = np.mat(array)
    seqs.append(list(array))#不断在尾部添加序列
    mydict[i] = re.findall(".*_(.*).txt.*",path)[0]#用字典存储列表索引和对应翻译,正则取出路径中的关键词'''
#f = open('mydic.txt','w')
#f.write(str(mydict))
#f.close
#seqs_array = np.array(seqs)
#np.save('muban.npy',seqs)
#print(mydict)
seqs = np.load('muban.npy',allow_pickle=True)
seqs = seqs.tolist()
print(seqs)
f = open('mydic.txt','r')
a = f.read()
mydict = eval(a)
f.close
seqs2 = []#空列表存储dtw距离

start = time.process_time()
for i in seqs:
    i = np.mat(i)
    cost= compute_distances_no_loops(xlist,i,trace = False)
    print(cost)
    seqs2.append(cost)
print('-------------------')
#第一第二匹配结果比较防止误判
min_cost = min(seqs2)
second_min_cost = second_min(seqs2)

min_index = seqs2.index(min_cost)
second_min_index = seqs2.index(second_min_cost)

min_matrix = np.mat(seqs[min_index])
second_min_matrix = np.mat(seqs[second_min_index])

_,path_list1 = compute_distances_no_loops(xlist,min_matrix,trace = True)#输出路径索引
_,path_list2 = compute_distances_no_loops(xlist,second_min_matrix,trace = True)

fenliScore1,match_cost1 = fenli(path_list1)
fenliScore2,match_cost2 = fenli(path_list2)

print(fenliScore1,match_cost1)
print(fenliScore2,match_cost2)

print(path_list1,path_list2)
#防误判规则：第一匹配结果与第二匹配结果比值十倍以上 or 分粒匹配分数大于0.8并小于均值的cost第一匹配结果与第二匹配结果比值十倍以上
if(min_cost/second_min_cost < 0.1 or (fenliScore1 >= 0.8 and match_cost1/match_cost2 <0.1)):
    print('result:',mydict[seqs2.index(min(seqs2))])
    print('second_result',mydict[seqs2.index(second_min_cost)])
    print('min_cost:',min_cost)
    print('second_cost:',second_min_cost)
    #print(min_matrix)
else:
    print('No Result!')
    print('min_cost:',min_cost)
    print('second_cost:',second_min_cost)

end = time.process_time()
time = end -start
print(time)

'''
text1 = np.loadtxt('/home/xuyt/文档/手语识别/处理后/_爱.txt')
#text1 = text1.flatten()
text2 = np.loadtxt('/home/xuyt/文档/手语识别/处理后/_把.txt')
#text2 = text2.flatten()
#print(text1)
#print(text2)
plt.plot(text1, label='Test1')
plt.plot(text2, label='Test2')
plt.legend()
#l2_norm = lambda x, y: (x - y) ** 2
#cost, _,_,path = accelerated_dtw(text1,text2,dist = l2_norm)
#dist = EuclideanDistances(text1,text2)
#print(dist)
print(text1)
plt.show()'''
