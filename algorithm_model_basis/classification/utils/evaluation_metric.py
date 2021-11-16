# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/4/8 14:29 
# 文件     ：evaluation_metric.py
# IDE     : PyCharm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from utils.config import *

class Evaluation(object):
    def __init__(self,true,pred):
        self.true = true
        self.pred = pred
    def multiclass_logloss(self, eps=1e-15):
        """对数损失度量（Logarithmic Loss  Metric）的多分类版本。
        :param actual: 包含actual target classes的数组
        :param predicted: 分类预测结果矩阵, 每个类别都有一个概率
        """
        # Convert 'actual' to a binary array if it's not already:
        if len(self.true.shape) == 1:
            actual2 = np.zeros((self.true.shape[0], self.pred.shape[1]))
            for i, val in enumerate(self.true):
                actual2[i, val] = 1
            self.true = actual2

        clip = np.clip(self.pred, eps, 1 - eps)
        rows = self.true.shape[0]
        vsota = np.sum(self.true * np.log(clip))
        return -1.0 / rows * vsota
    def acc(self):
        '''
        :return: 准确率评估
        '''
        acc = accuracy_score(np.argmax(self.true,axis = 1), np.argmax(self.pred,axis = 1))
        return acc
    def kappa(self):
        '''
        kappa系数是用在统计学中评估一致性的一种方法，取值范围是[-1,1]，实际应用中，一般是[0,1]，与ROC曲线中一般不会出现下凸形曲线的原理类似。这个系数的值越高，则代表模型实现的分类准确度越高。
        '''
        kappa = cohen_kappa_score(np.argmax(self.true,axis = 1), np.argmax(self.pred,axis = 1),labels=None)  # (label除非是你想计算其中的分类子集的kappa系数，否则不需要设置)
        return kappa
    def classificaiton_report(self):
        '''
        :return:
        '''
        if len(np.unique(np.argmax(self.true,axis = 1))) == config.numclass:
            target_names = ['class {:d}'.format(i) for i in np.arange(config.numclass)]
        else:
            target_names = ['class {:d}'.format(i) for i in np.unique(np.argmax(self.true,axis = 1)) ]
        return classification_report(np.argmax(self.true, axis=1), np.argmax(self.pred, axis=1),target_names=target_names, digits=4)