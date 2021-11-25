#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# 创建模型得第一种方式
model = Sequential([
	Dense(32,input_shape=(784,)),
	Activation('relu'),
	Dense(10),
	Activation('softmax')
])
# 创建模型得第二种方式
model = Sequential()
model.add(Dense(32, input_dim=784)) # 等价于 model.add(Dense(32,input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

# 模型编译
'''
complie方法,接受三个参数[优化器optimizer, 损失函数loss function, 评估标准matrices]
'''
model.compile(
	optimizer = 'rmsprop',
	loss = 'categorical_crossentropy',
	metrics = ['accuracy']
)
# 自定义评估函数
import tensorflow.keras.backend as K
def mean_pred(y_true, y_pred):
	return K.mean(y_true, y_pred)
model.compile(
	optimizer = 'rmsprop',
	loss = 'binary_crossentropy',
	metrics = ['accuracy', mean_pred]
)


# 模型训练
'''
Keras 模型在输入数据和标签的 Numpy 矩阵上进行训练。为了训练一个模型，你通常会使用 fit 函数。
'''
import numpy as np
data = np.random.random((1000,100))
label = np.random.randint(2, size=(1000, 1))
model.fit(data, label, epochs = 10, batch_size = 32)