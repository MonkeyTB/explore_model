# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2022/4/21 17:00 
# 文件     ：knowlege_distiller.py
# IDE     : PyCharm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
        '''
        :optimizer: Keras优化器为学生权重
        :metrics: Keras评估指标
        :student_loss_fn: 学生预测与真实之间差异的损失函数 hard_loss
        :distillation_loss_fn: 学生预测和教师预测之间差异的损失函数 soft loss
        :alpha:
        :temperature: 软化概率分布的 T,较大的T给出了更均匀的分布
        '''
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
    def train_step(self, data):
        x, y = data
        # 教师模型的前向
        teacher_predictions = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            # 学生模型的前向
            student_predicitons = self.student(x, training=True)

            # loss
            student_loss = self.student_loss_fn(y, student_predicitons)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis = 1),
                tf.nn.softmax(student_predicitons / self.temperature, axis = 1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        # 计算梯度
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # 更新权重
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # 更新在“编译（）`中配置的指标
        self.compiled_metrics.update_state(y, student_predicitons)

        # 返回表现字典
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {'student_loss':student_loss, 'distillation_loss':distillation_loss}
        )
        return results
    def test_step(self, data):
        x, y = data

        #
        y_prediction = self.student(x, training=False)

        # 计算loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # 更新指标
        self.compiled_metrics.update_state(y, y_prediction)

        #
        results = {m.name:m.result() for m in self.metrics}
        results.update({"student_loss":student_loss})
        return results
teacher = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(10),
    ], name='tearcher'
)
student = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(10),
    ], name='student'
)
# clone 一个
student_scratch = keras.models.clone_model(student)


batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Normalize data
x_train = x_train.astype("float32") / 255.0
x_train = np.reshape(x_train, (-1, 28, 28, 1))

x_test = x_test.astype("float32") / 255.0
x_test = np.reshape(x_test, (-1, 28, 28, 1))

# 训练老师
teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
teacher.fit(x_train, y_train, epochs=5)
# teacher.evaluate(x_test, y_test)
print('KD')
# 从老师到学生，训练
# 初始化和编译蒸馏器
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)
# 蒸馏教师和学生
distiller.fit(x_train, y_train, epochs=3)
distiller.evaluate(x_test, y_test)
print('clone')
# Train student as doen usually
student_scratch.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train and evaluate student trained from scratch.
student_scratch.fit(x_train, y_train, epochs=3)
student_scratch.evaluate(x_test, y_test)