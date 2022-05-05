# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2022/2/15 10:18 
# 文件     ：loss.py
# IDE     : PyCharm
'''
focal loss: 主要为了解决样本不平衡问题
'''
import os
from keras import backend as K
import tensorflow as tf


def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


def multi_category_focal_loss1(alpha, gamma=2.0):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha用于指定不同类别/标签的权重，数组大小需要与类别个数一致
    当你的数据集不同类别/标签之间存在偏斜，可以尝试适用本函数作为loss
    Usage:
     model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    alpha = tf.constant(alpha, dtype=tf.float32)
    # alpha = tf.constant([[1],[1],[1],[1],[1]], dtype=tf.float32)
    # alpha = tf.constant_initializer(alpha)
    gamma = float(gamma)

    def multi_category_focal_loss1_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss

    return multi_category_focal_loss1_fixed


def multi_category_focal_loss2(gamma=2., alpha=.25):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha控制真值y_true为1/0时的权重
        1的权重为alpha, 0的权重为1-alpha
    当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
    当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
    当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)
        尝试将alpha调大,鼓励模型进行预测出1。
    Usage:
     model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss

    return multi_category_focal_loss2_fixed


class multi_focal_loss2(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(multi_focal_loss2, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0)
        y_true = tf.cast(y_true, tf.float32)

        loss = -y_true * tf.math.pow(1 - y_pred, self.gamma) * tf.math.log(y_pred)
        loss = tf.reduce_sum(loss, axis=1)
        return loss


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1.0)
    y_true = tf.cast(y_true, tf.float32)

    loss = -y_true * tf.math.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
    loss = tf.reduce_sum(loss, axis=1)
    return loss


# 修正的交叉熵损失
margin = 0.6
theta = lambda t : (K.sign(t) + 1.) / 2
def bes_loss(y_true, y_pred):
    return - (1 - theta(y_true - margin) * theta(y_pred - margin)
            - theta(1 - margin - y_true) * theta(1 - margin - y_pred)
         ) * (y_true * K.log(y_pred + 1e-8) + (1 - y_true) * K.log(1 - y_pred + 1e-8))