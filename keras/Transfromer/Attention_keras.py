# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/11/2 19:18 
# 文件     ：Attention_keras.py
# IDE     : PyCharm
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class Position_Embedding(Layer):
    def __init__(self, size=None, mode = 'sum', **kwargs):
        self.size = size # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size':self.size,
            'mode':self.mode})
        return config
    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_lengh = K.shape(x)[0], K.shape(x)[1]
        position_j = K.arange(self.size,dtype='float32')

        position_j = K.expand_dims(position_j, 0)

        # position_i = K.cumsum(K.ones_like(x[:,:,0]),1)-1
        # position_i = K.expand_dims(position_i, 2)

        # position_ij = K.dot(position_i, position_j)
        # position_ij = K.concatenate([K.cos(position_ij),K.sin(position_ij)], 2)

        if self.mode == 'sum':
            return position_j + x
        elif self.mode == 'concat':
            return K.concatenate([position_j, x], 2)
    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.size)

class MultiHeadAttention(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(MultiHeadAttention, self).__init__(**kwargs)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'nb_head':self.nb_head,
            'size_per_head':self.size_per_head
        })
        return config
    def build(self, input_shape): # input_shape  ==  x [TensorShape([None, None, 128]),TensorShape([None, None, 128]),TensorShape([None, None, 128])]
        self.WQ = self.add_weight(name='WQ', shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform', trainable = True) # (128, 128)
        self.WK = self.add_weight(name='WK', shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform', trainable = True)
        self.WV = self.add_weight(name='WV', shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform', trainable = True)
        super(MultiHeadAttention, self).build(input_shape)
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) *1e12
    def call(self, x):
        '''
        :param x: 只传入Q_seq,K_seq,V_seq,那么就不做Mask
                  同时传入Q_seq,K_seq,V_seq,Q_len,V_len,那么对多余部分做Mask
        :return:
        '''
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ) # 矩阵相乘 （None, None, 128）*(128, 128) -> (None, None, 128)
        Q_seq = K.reshape(Q_seq, (-1,K.shape(Q_seq)[1], self.nb_head, self.size_per_head)) # (None, None, 8, 16)
        Q_seq = K.permute_dimensions(Q_seq,(0,2,1,3)) # (0,1,2,3)-->(0,2,1,3) (None, 8, None, 16)

        K_seq = K.dot(K_seq, self.WK) # 矩阵相乘
        K_seq = K.reshape(K_seq, (-1,K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq,(0,2,1,3)) # (0,1,2,3)-->(0,2,1,3)

        V_seq = K.dot(V_seq, self.WK)  # 矩阵相乘
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))  # (0,1,2,3)-->(0,2,1,3)

        # 计算内积，然后mask，然后softmax
        '''
        https://blog.csdn.net/weixin_48384960/article/details/108862037
        内置keras.backend的batch_dot函数会出现维度溢出的现象，两个四阶张量计算结果为五阶张量
        example:
            # 三维
            x_batch = tf.keras.backend.ones(shape=(32,20,1))
            y_batch = tf.keras.backend.ones(shape=(32,30,20))
            xy_batch_dot = tf.keras.backend.batch_dot(x_batch,y_batch,axes=(1,2))
            xy_batch_dot.shape --> TensorShape([32, 1, 30])
            # 四维
            x_batch = tf.keras.backend.ones(shape=(32,20,1,4))
            y_batch = tf.keras.backend.ones(shape=(32,30,20,4))
            xy_batch_dot = tf.keras.backend.batch_dot(x_batch,y_batch,axes=(1,2))
            xy_batch_dot.shape --> TensorShape([32, 1, 4, 30, 4])
        '''
        # A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head ** 0.5
        Q_seq_reshape = K.reshape(Q_seq, (-1, K.shape(Q_seq)[2],K.shape(Q_seq)[3])) # (None, 8, None, 16) -> (None, None, 16)
        K_seq_reshape = K.reshape(K_seq, (-1, K.shape(K_seq)[2],K.shape(K_seq)[3]))
        A = K.batch_dot(Q_seq_reshape, K_seq_reshape, axes=[2,2]) / self.size_per_head ** 0.5 # (None, None, None)
        A = K.reshape(A,(-1, K.shape(Q_seq)[1], K.shape(A)[1], K.shape(A)[2])) # (1, None, None, None)

        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add') # (None, None, None, None)
        A = K.permute_dimensions(A,(0,3,2,1))
        A = K.softmax(A)

        # 输出并mask
        # O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        A_reshape = K.reshape(A, (-1,K.shape(A)[2],K.shape(A)[3]))
        V_seq_reshape = K.reshape(V_seq, (-1,K.shape(V_seq)[2],K.shape(V_seq)[3]))
        Z = K.batch_dot(A_reshape, V_seq_reshape, axes=[2,1])
        Z = K.reshape(Z, (-1, K.shape(A)[1], K.shape(Z)[1], K.shape(Z)[2]))

        Z = K.permute_dimensions(Z,(0,2,1,3))
        Z = K.reshape(Z, (-1,K.shape(Z)[1], self.output_dim))
        Z = self.Mask(Z, Q_len, 'mul')
        return Z
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1],self.output_dim)
