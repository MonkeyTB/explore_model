#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
from tensorflow.keras.layers import Input, Dense, Flatten, Masking, RNN
from tensorflow.keras.models import Sequential, Model
layer = Dense(32)
config = layer.get_config()
print( config ) # 以含有Numpy矩阵的列表形式返回层的权重
print( layer.get_weights() ) # 以含有Numpy矩阵的列表形式返回层的权重
print( Dense.from_config(config))

# 2. Dense
'''
Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
input: nD张量，尺寸（batch_size, ..., input_dim）
output:nD张量，尺寸（batch_size, ..., units）
'''

# 3. Fallent
'''
keras.layers.Flatten(data_format=None)
data_format:一个字符串，其值为channel_last（默认）或者channel_first。它表明输入的维度的顺序。此参数的目的是为了当模型
从一个数据格式切换到另一个数据格式时保留权重顺序。
'''
model = Sequential()
model.add(Conv2D(64, (3, 3),
                 input_shape=(3, 32, 32), padding='same',))
# 现在：model.output_shape == (None, 64, 32, 32)
model.add(Flatten())
# 现在：model.output_shape == (None, 65536)

# 4. Permute
'''
keras.layers.Permute(dims)
根据给定的模式置换输入的维度
dims：整数元组，置换模式，不包含样本的维度。索引从1开始，例如（2，1），就是置换第一维和第二维
输入尺寸：任意
输出尺寸：输入尺寸

model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
# 现在： model.output_shape == (None, 64, 10)
# 注意： `None` 是批表示的维度
'''
# 5. RepeatVector
'''
keras.layers.RepeatVector(n)
:将重复n次
n：整数，输入的次数
输入：2D张量
输出：3D张量
model = Sequential()
model.add(Dense(32, input_dim=32))
# 现在： model.output_shape == (None, 32)
# 注意： `None` 是批表示的维度
model.add(RepeatVector(3))
# 现在： model.output_shape == (None, 3, 32)
'''

# 6. Lambda
'''
keras.layers.Lambda(function, output_shape=None, mask=None, arguments=None)
将任意表达式封装成layer对象
function: 需要封装的函数。 将输入张量作为第一个参数。
输入尺寸:任意
输出尺寸:由 output_shape 参数指定 (或者在使用 TensorFlow 时，自动推理得到)。
# 添加一个 x -> x^2 层
model.add(Lambda x : x**2)

# 添加一个网络层，返回输入的正数部分
# 与负数部分的反面的连接

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] *= 2
    return tuple(shape)
model.add(Lambda(antirectifier,
                 output_shape=antirectifier_output_shape))
'''

# 7. Masking
'''
keras.layers.Masking(mask_value=0.0)
使用覆盖值覆盖序列,以跳过时间步
对于输入张量的每一个时间步(张量的第一个维度),如果所有时间步输入张量的值与mask_value相等,那么这个时间步将再下游层被
覆盖(跳过)(只要他们支持覆盖)
不支持覆盖则会抛出异常

例:
考虑将要喂入一个 LSTM 层的 Numpy 矩阵 x， 尺寸为 (samples, timesteps, features)。
你想要覆盖时间步 #3 和 #5，因为你缺乏这几个 时间步的数据。你可以：

设置 x[:, 3, :] = 0. 以及 x[:, 5, :] = 0.
在 LSTM 层之前，插入一个 mask_value=0 的 Masking 层：
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32))
'''

# 8. keras实现自定义层
'''
对于简单的无定义的自定义操作,可以通过keras.core.Lambda层实现,对于包含可训练权重的自定义层,应该自己实现这些层
需要实现三个方法
1. build(input_shape):这是定义权重的地方,这个方法必须设self.build = True,可以通过调用super(Layer,self).build()
2. call(x):这里编写层的功能逻辑的地方,需要关注传入call的第一个参数:输入张量,除非你希望你的层支持masking
3. compute_output_shape(input_shape): 如果你的层更改了输入张量的形状，你应该在这里定义形状变化的逻辑，这让Keras能够自动推断各层的形状。
'''
from keras import backend as K
from keras.engine.topology import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

'''
还可以定义具有多个输入张量和多个输出张量的 Keras 层。 
为此，你应该假设方法 build(input_shape)，call(x) 和 compute_output_shape(input_shape) 的输入输出都是列表。 
这里是一个例子，与上面那个相似：
'''
from keras import backend as K
from keras.engine.topology import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]