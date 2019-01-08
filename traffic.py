"""
Author: deepinwst
Email: wanshitao@donews.com
Date: 19-1-5 下午4:15
"""

import numpy as np
from random import random as rand
from utils import SmoothedLastSlotReqs, SmoothedNDayReqs
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = 'Yahei Mono'


# 选择最近K天的流量为基准
k = 15;             # 生成的请求的数据
tol = 400;          # 最大容忍度
trainK = 7;         # 训练样本的天数
T = 24;             # 每天的时间片数
fakeSmooth = 4;
lastNDayRequest = np.zeros([trainK, T]);
lastSlotReqs = np.zeros([trainK, T]);
trueValue = np.zeros([trainK, T]);

reqs = np.zeros([k, T]);

for i in range(k):
    reqs[i][0] = 23 + rand();
    reqs[i][1] = 12.445 + rand() * 10.445 / fakeSmooth
    reqs[i][2] = 6.454 + rand() * 4.454 / fakeSmooth
    reqs[i][3] = 3.2155 + rand() * 2.215 / fakeSmooth
    reqs[i][4] = 2.394 + rand() * 1.394 / fakeSmooth
    reqs[i][5] = 3.273 + rand() * 2.273 / fakeSmooth
    reqs[i][6] = 6.225 + rand() * 5.225 / fakeSmooth
    reqs[i][7] = 11.523 + rand() * 11.523 / fakeSmooth
    reqs[i][8] = 15.911 + rand() * 15.911 / fakeSmooth
    reqs[i][9] = 16.839 + rand() * 16.839 / fakeSmooth
    reqs[i][10] = 16.414 + rand() * 16.414 / fakeSmooth
    reqs[i][11] = 16.808 + rand() * 16.808 / fakeSmooth
    reqs[i][12] = 19.598 + rand() * 19.598 / fakeSmooth
    reqs[i][13] = 15.036 + rand() * 15.036 / fakeSmooth
    reqs[i][14] = 15.204 + rand() * 15.204 / fakeSmooth
    reqs[i][15] = 19.085 + rand() * 19.085 / fakeSmooth
    reqs[i][16] = 18.855 + rand() * 18.855 / fakeSmooth
    reqs[i][17] = 19.520 + rand() * 19.520 / fakeSmooth
    reqs[i][18] = 19.713 + rand() * 19.713 / fakeSmooth
    reqs[i][19] = 21.547 + rand() * 21.547 / fakeSmooth
    reqs[i][20] = 23.107 + rand() * 3.107 / fakeSmooth
    reqs[i][21] = 25.534 + rand() * 25.534 / fakeSmooth
    reqs[i][22] = 24.329 + rand() * 24.329 / fakeSmooth
    reqs[i][23] = 22 + rand() * 22 / fakeSmooth;

# 获取到训练数据, 需要估计的参数有27个，训练样本有24 * 7 = 168个
X = np.zeros([trainK * T, 27])
Y = np.zeros([trainK * T, 1])
for i in range(trainK):
    for j in range(T):
        # SmoothedNDayReqs
        lastNDayRequest[i, j] = SmoothedNDayReqs(reqs, j, trainK, i)()
        # SmoothedLastSlotReqs
        lastSlotReqs[i][j] = SmoothedLastSlotReqs(reqs, j, trainK, tol, i)()
        trueValue[i][j] = reqs[i][j]
        w = np.zeros([1, T])
        w[0][j] = 1
        X[i * T + j,:] =np.append(w, [lastNDayRequest[i, j], lastSlotReqs[i, j], 1])
        Y[i * T + j, 0] = trueValue[i, j]

# LR, r为残差，rint置信区间，
regression = LinearRegression()
regression.fit(X, Y)
predict = regression.predict(X)

x = np.zeros([1 * T, 27])
y = np.zeros([1 * T, 1])
for i in range(1):
    for j in range(T):
        w = np.zeros([1, T])
        w[0][j] = 1
        x[i * T + j, :] = np.append(w, [lastNDayRequest[i, j], lastSlotReqs[i, j], 1])
        y[i * T + j, 0] = trueValue[i, j]

res = regression.predict(x)
print("x:", x)
print("res:", res)
plt.plot(np.arange(24), res, 'r--', label='预测流量')
plt.plot(np.arange(24), y, 'b-', label='真实流量')
plt.legend(loc='best')
plt.xlabel('时段');
plt.ylabel('流量');
plt.show()
