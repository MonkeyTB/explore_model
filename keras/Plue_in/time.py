# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/12/21 16:08 
# 文件     ：time.py
# IDE     : PyCharm
import time
import numpy as np
class Time:
    '''记录多次运行时间'''
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        '''启动计时器'''
        self.tik = time.time()
    def stop(self):
        '''停止计时器并将时间记录再列表中'''
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def avg(self):
        '''返回平均时间'''
        return sum(self.times) / len(self.times)
    def sum(self):
        '''返回时间总合'''
        return sum(self.times)
    def cumsum(self):
        '''返回累计时间'''
        return np.array(self.times).cumsum().tolist()

if __name__ == '__main__':
    timer = Time()
    sum = 0
    for i in range(10000):
        sum += i
    print('{} sec'.format(timer.stop()))
    print('{} sec'.format(timer.cumsum()))
