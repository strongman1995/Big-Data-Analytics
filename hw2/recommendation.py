
# coding: utf-8

# ## project 2 个性化推荐
# ### 数据预处理

import numpy as np
import datetime
from sklearn import preprocessing
import numpy.linalg as LA
import matplotlib.pyplot as plt


user_dict = {} # user_id: matrix_index
with open('Project2-data/users.txt') as f:
    user_data = f.readlines()
    for i in range(len(user_data)):
        user_data[i] = user_data[i].strip()
        user_dict[user_data[i]] = i


train = np.empty([10000,10000],dtype=np.int32)
with open('Project2-data/netflix_train.txt') as f:
    lines = f.readlines()
    for i,line in enumerate(lines):
        if i % 1000000 == 0:
            print i
        record = line.strip().split(' ') # user_id movie_id,score,date
        user_id,movie_id,score = record[0],record[1],int(record[2])
        user_index = user_dict[user_id]
        movie_index = int(movie_id)-1
        train[user_index][movie_index] = score
print "import train data successfully"


test = np.empty([10000,10000],dtype=np.int32)
with open('Project2-data/netflix_test.txt') as f:
    lines = f.readlines()
    for i,line in enumerate(lines):
        if i % 1000000 == 0:
            print i
        record = line.strip().split(' ') # user_id movie_id,score,date
        user_id,movie_id,score = record[0],record[1],int(record[2])
        user_index = user_dict[user_id]
        movie_index = int(movie_id)-1
        test[user_index][movie_index] = score
test_len = i+1
print "import test data successfully"


begin = datetime.datetime.now() # 开始计时

train_normalized = preprocessing.normalize(train, norm='l2') # 对train集做归一化
sim = train_normalized.dot(train_normalized.T) # 得到相似度矩阵sim(i,j)=train集用户i对train集用户j的cos相似度

numerator = sim.dot(train)  # 分子
denominator = sim.dot(np.ones((10000,10000)))  # 分母
score = numerator/denominator # 预测分数

flag_test = test
flag_test[flag_test > 0] = 1

# RMSE = (np.sum((score*flag_test - test)**2)/test_len)**0.5 # 只考虑test集中有评价的那些打分
RMSE = LA.norm(score*flag_test - test,"fro")/(test_len**0.5) # 均方根误差
print "RMSE:",RMSE
print "运行时间：%d s"%((datetime.datetime.now() - begin).seconds) # 计时结束


# ### 基于梯度下降的矩阵分解算法

# 生成参数集
k_list = [20,50]
lambda_list = [0.001,0.01,0.1]
parameter_list = []
for k in k_list:
    for lbd in lambda_list:
        parameter_list.append((k,lbd))


iters = []
Js = []
RMSEs = []
Losses = []

with open('log1.txt','w') as f:
    for idx,(k,lambda_) in enumerate(parameter_list):
        f.write("==========================\n")
        f.write("parameter %d: k: %d\tlambda: %f\n"%(idx+1,k,lambda_))
        f.write("==========================\n")
        # 初始化
        alpha = 0.0001
        L = 0.00005
        U = 0.01*np.random.rand(10000,k)
        V = 0.01*np.random.rand(10000,k)
        X = train
        A = train
        A[A > 0] = 1
        B = test
        B[B > 0] = 1
        
        J = 0.5*(LA.norm(A*(X-U.dot(V.T)),"fro")**2) + lambda_*(LA.norm(U,"fro")**2) + lambda_*(LA.norm(V,"fro")**2)
        J = J/1e7
        Jo = J+1
        iter_list = []
        J_list = []
        RMSE_list = []
        Loss_list = []
        iter_ = 1
        begin = datetime.datetime.now()
        # 迭代进行梯度下降
        while Jo - J > L and iter_ < 200:
            Jo = J
            
            J_U = (A*(U.dot(V.T)-X)).dot(V) + 2*lambda_*U
            J_V = (A*(U.dot(V.T)-X)).dot(U) + 2*lambda_*V
            
            U = U - alpha * J_U
            V = V - alpha * J_V
            
            J = 0.5*(LA.norm(A*(X-U.dot(V.T)),"fro")**2) + lambda_*(LA.norm(U,"fro")**2) + lambda_*(LA.norm(V,"fro")**2)
            J = J/1e7
            
            RMSE = LA.norm(B*U.dot(V.T) - test,"fro")/(test_len**0.5)
            
            loss = Jo - J
            
            f.write("iter: %d \t J: %.6f \t RMSE: %.6f \t Loss: %.6f\n"%(iter_,J,RMSE,loss))
            
            iter_list.append(iter_)
            J_list.append(J)
            RMSE_list.append(RMSE)
            Loss_list.append(loss)
            
            iter_ += 1
    
        iters.append(iter_list)
        Js.append(J_list)
        RMSEs.append(RMSE_list)
        Losses.append(Loss_list)
        
        f.write("========= end =========\n")
        f.write("iter: %d \t J: %.6f \t RMSE: %.6f \t Loss: %.6f\n"%(iter_,J,RMSE,loss))
        f.write("运行时间：%d s\n"%((datetime.datetime.now() - begin).seconds))
        # 画图
        plt.subplot(121)
        plt.title("J. k:%d  lambda:%f"%(k,lambda_))
        plt.xlabel("iter")
        plt.ylabel('J')
        plt.plot(iter_list, J_list, 'b')
        
        plt.subplot(122)
        plt.title("RMSE. k:%d  lambda:%f"%(k,lambda_))
        plt.xlabel("iter")
        plt.ylabel('RMSE')
        plt.plot(iter_list, RMSE_list, 'r')
        plt.figure(1).tight_layout()
        plt.savefig('pic/'+str(k)+'_'+str(lambda_)+'.png')
        plt.cla()
        
        plt.subplot(111)
        plt.title("J-RMSE. k:%d  lambda:%f"%(k,lambda_))
        plt.xlabel("J")
        plt.ylabel('RMSE')
        plt.plot(J_list, RMSE_list, 'g')
        plt.figure(1).tight_layout()
    plt.savefig('pic/J_RMSE_'+str(k)+'_'+str(lambda_)+'.png')


import pylab as pl
colors = ['b','g','r','y','m','c']
legends = ['k-%d lbd-%f'%(k,lambda_) for k,lambda_ in parameter_list]


for idx in [0,3]:
    (k,lambda_)=parameter_list[idx]
    iter_list = iters[idx]
    J_list = Js[idx]
    RMSE_list = RMSEs[idx]
    Loss_list = Losses[idx]
    color = colors[idx]
    label = legends[idx]
    plot = pl.plot(iter_list,J_list,color=color,label=label)

pl.title('J')
pl.xlabel('iter')
pl.ylabel('J')
pl.legend(loc='best')# make legend
pl.show()


for idx in [0,3]:
    (k,lambda_)=parameter_list[idx]
    iter_list = iters[idx]
    J_list = Js[idx]
    RMSE_list = RMSEs[idx]
    Loss_list = Losses[idx]
    color = colors[idx]
    label = legends[idx]
    plot = pl.plot(iter_list,RMSE_list,color=color,label=label)

pl.title('RMSE')
pl.xlabel('iter')
pl.ylabel('RMSE')
pl.legend(loc='best')# make legend
pl.show()


for idx,(k,lambda_) in enumerate(parameter_list):
    iter_list = iters[idx]
    J_list = Js[idx]
    RMSE_list = RMSEs[idx]
    Loss_list = Losses[idx]
    color = colors[idx]
    label = legends[idx]
    plot = pl.plot(J_list,RMSE_list,color=color,label=label)

pl.title('J-RMSE')
pl.xlabel('J')
pl.ylabel('RMSE')
pl.legend(loc='best')# make legend
pl.show()

