#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

# 1.如何在GPU上运行keras
'''
如果你以 TensorFlow 或 CNTK 后端运行，只要检测到任何可用的 GPU，那么代码将自动在 GPU 上运行。
如果你以 Theano 后端运行，则可以使用以下方法之一：
方法 1: 使用 Theano flags。

THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py
"gpu" 可能需要根据你的设备标识符（例如gpu0，gpu1等）进行更改。

方法 2: 创建 .theanorc: 指导教程

方法 3: 在代码的开头手动设置 theano.config.device, theano.config.floatX：

import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
'''

# 2. 如何在多 GPU 上运行 Keras 模型?
'''
我们建议使用 TensorFlow 后端来执行这项任务。有两种方法可在多个 GPU 上运行单个模型：数据并行和设备并行。
数据并行
	每个设备上复制一份数据,并在不同得设备上处理不同得输入数据部分, 内置函数可以实现:keras.utils.multi_gpu_model,
	可以生成数据得并行版本,在多达8个GPU上实现准线性加速
'''
from tensorflow.keras.utils import  multi_gpu_model
# 将 `model` 复制到 8 个 GPU 上。
# 假定你的机器有 8 个可用的 GPU。
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')
parallel_model.fit(x, y, epochs=20, batch_size=64)
'''
设备并行
	设备并行包括在不同得设备上运行同意模型得不同部分,对于具有并行体系结构的模型，例如有两个分支的模型，这种方式很合适。
'''
from tensorflow.keras.layers import Input, LSTM, concatenate
import tensorflow as tf
# 模型中共享的 LSTM 用于并行编码两个不同的序列
input_a = Input(shape=(140, 256))
input_b = Input(shape=(140, 256))

shared_lstm = LSTM(64)

# 在一个 GPU 上处理第一个序列
with tf.device_scope('/gpu:0'):
    encoded_a = shared_lstm(tweet_a)
# 在另一个 GPU上 处理下一个序列
with tf.device_scope('/gpu:1'):
    encoded_b = shared_lstm(tweet_b)

# 在 CPU 上连接结果
with tf.device_scope('/cpu:0'):
    merged_vector = concatenate([encoded_a, encoded_b], axis=-1)

# 3. "sample", "batch", "epoch" 分别是什么？
'''
为了正确地使用 Keras，以下是必须了解和理解的一些常见定义：

Sample: 样本，数据集中的一个元素，一条数据。
例1: 在卷积神经网络中，一张图像是一个样本。
例2: 在语音识别模型中，一段音频是一个样本。
Batch: 批，含有 N 个样本的集合。每一个 batch 的样本都是独立并行处理的。在训练时，一个 batch 的结果只会用来更新一次模型。
一个 batch 的样本通常比单个输入更接近于总体输入数据的分布，batch 越大就越近似。然而，每个 batch 将花费更长的时间来处理，并且仍然只更新模型一次。在推理（评估/预测）时，建议条件允许的情况下选择一个尽可能大的 batch，（因为较大的 batch 通常评估/预测的速度会更快）。
Epoch: 轮次，通常被定义为 「在整个数据集上的一轮迭代」，用于训练的不同的阶段，这有利于记录和定期评估。
当在 Keras 模型的 fit 方法中使用 validation_data 或 validation_split 时，评估将在每个 epoch 结束时运行。
在 Keras 中，可以添加专门的用于在 epoch 结束时运行的 callbacks 回调。例如学习率变化和模型检查点（保存）。
'''
# 4. 如何保存keras模型
'''
使用 model.save(filepath) 将 Keras 模型保存到单个 HDF5 文件中，该文件将包含：
	模型的结构，允许重新创建模型
	模型的权重
	训练配置项（损失函数，优化器）
	优化器状态，允许准确地从你上次结束的地方继续训练
使用 keras.models.load_model(filepath) 重新实例化模型。load_model 还将负责使用
保存的训练配置项来编译模型（除非模型从未编译过）
'''
from tensorflow.keras.models import save_model, load_model
model.save_model('my.h5')
del model
model = load_model('my.h5')

# 5. 只保存/加载模型结构
# 保存为 JSON
json_string = model.to_json()
# 保存为 YAML
yaml_string = model.to_yaml()

# 从 JSON 重建模型：
from tensorflow.keras.models import model_from_json
model = model_from_json(json_string)

# 从 YAML 重建模型：
from tensorflow.keras.models import model_from_yaml
model = model_from_yaml(yaml_string)

# 6. 只保存/加载模型的权重
model.save_weights('my_model_weights.h5')
# 假设你有用于实例化模型的代码，则可以将保存的权重加载到具有相同结构的模型中：
model.load_weights('my_model_weights.h5')
# 如果你需要将权重加载到不同的结构（有一些共同层）的模型中，例如微调或迁移学习，则可以按层的名字来加载权重：
model.load_weights('my_model_weights.h5', by_name=True)

"""
假设原始模型如下所示：
    model = Sequential()
    model.add(Dense(2, input_dim=3, name='dense_1'))
    model.add(Dense(3, name='dense_2'))
    ...
    model.save_weights(fname)
"""

# 新模型
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # 将被加载
model.add(Dense(10, name='new_dense'))  # 将不被加载

# 从第一个模型加载权重；只会影响第一层，dense_1
model.load_weights(fname, by_name=True)
# 7.处理已保存模型中的自定义层（或其他自定义对象）
'''
如果要加载的模型包含自定义层或其他自定义类或函数，则可以通过 custom_objects 参数将它们传递给加载机制：
'''
from tensorflow.keras.models import  load_model
# 假设你的模型包含一个 AttentionLayer 类的实例
model = load_model('my_model.h5', custom_objects={'AttentionLayer':AttentionLayer})

'''或者，你可以使用 自定义对象作用域：'''
from tensorflow.keras.utils import CustomObjectScope
with CustomObjectScope({'AttentionLayer':AttentionLayer}):
	model = load_model('my_model.h5')

# 8. 如何获取中间层得输出
from tensorflow.keras.models import Model
model = ...# 创建原始模型
layer_name = 'my_layer'
intermediate_layer_model = Model(inputs = model.input, outputs = model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
# 你也可以构建一个 Keras 函数，该函数将在给定输入的情况下返回某个层的输出，例如：
from tensorflow.keras import backend as K
# 以 Sequential 模型为例
get_3rd_layer_output = K.function([model.layers[0].input
								   model.layers[3].output])
layer_output = get_3rd_layer_output([x])[0]
# 9. 在验证集的误差不再下降时，如何中断训练？
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=2)
model.fit(x, y, callable=[early_stopping])

# 10. Model类继承

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
class SimpleMLP(Model):
	def __init__(self, use_bn = False, use_dp = False, num_class = 10):
		self.use_bn = use_bn
		self.use_dp = use_dp
		self.num_class = num_class

		self.dense1 = Dense(32, activation = 'relu')
		self.dense2 = Dense(num_class, activation = 'softmax')
		if self.use_dp:
			self.dp = Dropout(0.5)
		if self.use_bn:
			self.bn = BatchNormalization(axis = 1)
	def call(self, inputs):
		x = self.dense1(inputs)
		if self.use_dp:
			x = self.dp(x)
		if self.use_bn:
			x = self.bn
		return self.dense2(x)
model = SimpleMLP()
model.compile(...)
model.fit(...)
'''
网络层定义在 __init__(self, ...) 中，前向传播在 call(self, inputs) 中指定。
在 call 中，你可以指定自定义的损失函数，通过调用 self.add_loss(loss_tensor) 
（就像你在自定义层中一样）。
'''