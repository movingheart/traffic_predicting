"""
Author: deepinwst
Email: wanshitao@donews.com
Date: 19-1-4 下午6:23
"""
import numpy as np
import scipy


class SmoothedLastSlotReqs(object):
    def __init__(self, reqs, len, k, tol, start):
        """ 平滑上一个时间点的流量

        :param reqs: 最近n天的流量数据
        :param len: 时间点，如8点
        :param k: 训练集的天数
        :param tol: 最大容忍度
        :param start: 训练集中的第几天
        """
        self.reqs = reqs
        self.len = len
        self.k = k
        self.tol = tol
        self.start = start

    def __call__(self, *args, **kwargs):

        history = np.zeros([self.k, 1])
        if self.len == 1:
            originalLastReqs = self.reqs[self.start + 1, 23]
            for i in range(1, self.k):
                history[i] = self.reqs[self.start+i+1, 23]
        else:
            originalLastReqs = self.reqs[self.start, self.len - 1]
            for i in range(1, self.k):
                history[i] = self.reqs[self.start+i, self.len-1]
        avg = np.mean(history)
        std = np.std(history)

        if abs(originalLastReqs - avg) / std > self.tol:
            history[0] = originalLastReqs
            lastSlotReqs = scipy.stats.gmean(history)
        else:
            lastSlotReqs = originalLastReqs

        return lastSlotReqs


class SmoothedNDayReqs(object):
    def __init__(self, reqs, len, k, start):
        """ 过去N天的流量

        :param reqs: 最近n天的流量数据
        :param len: 时间点，如8点
        :param k: 训练集的天数
        :param start: 训练集中的第几天
        """
        self.reqs = reqs
        self.len = len
        self.k = k
        self.start = start

    def __call__(self, *args, **kwargs):
        history = np.zeros([self.k, 1])
        for i in range(self.k):
            history[i, 0] = self.reqs[self.start + i, self.len]

        return np.prod(history)**(1/len(history))


