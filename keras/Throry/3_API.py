#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

# 全连接层
'''
sequential模型是一个实现这种网络得一个很好的方式
输入输出都是张量
'''
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

inputs = Input(shape=(784,))
# 层的实例是可调用的，它以张量为参数，并且返回一个张量
x = Dense(64,activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
prediction = Dense(10, activation='softmax')(x)

# 这部分创建了一个包含输入层和三个全连接层的模型
model = Model(inputs=inputs, outputs=prediction)
model.compile(
	optimizer='rmsprop',
	loss='categorical_crossentropy',
	metrics=['accuracy']
)
model.fit(data, label)

#所有的模型都可调用，就像网络层一样
'''
利用函数式 API，可以轻易地重用训练好的模型：可以将任何模型看作是一个层，
然后通过传递一个张量来调用它。注意，在调用模型时，您不仅重用模型的结构，
还重用了它的权重。
'''
x = Input(shape=(784,))
y = model(x)
'''
这种方式能允许我们快速创建可以处理序列输入的模型。只需一行代码，你就将图像分类模型转换为视频分类模型。
'''
from tensorflow.keras.layers import TimeDistributed
input_sequences = Input(shape=(20, 784))
# 这部分将我们之前定义的模型应用于输入序列中的每个时间步。
# 之前定义的模型的输出是一个 10-way softmax，
# 因而下面的层的输出将是维度为 10 的 20 个向量的序列。
processed_sequences = TimeDistributed(model)(input_sequences)


# 多输入多输出
'''
https://keras.io/zh/getting-started/functional-api-guide/
'''
from tensorflow.keras.layers import Embedding, LSTM, concatenate
# 标题输入：接收一个含有 100 个整数的序列，每个整数在 1 到 10000 之间。
# 注意我们可以通过传递一个 "name" 参数来命名任何层。
main_input = Input(shape=(100,), type='int32', name='main_input')
# Embedding 层将输入序列编码为一个稠密向量的序列，
# 每个向量维度为 512
x = Embedding(output_dim=512, 	input_dim=10000, input_length=100)(main_input) # input_dim: int > 0. Size of the vocabulary,
lstm_out = LSTM(units=32)(x)
# 辅助输出
auxiliary_output = Dense(1, activation='sigmoid', name = 'aux_output')(lstm_out)
# 辅助输入
auxiliary_input = Input(shape=(5,), name='aux_input')
x = concatenate([lstm_out, auxiliary_input]) # lstm 得输出和辅助输入拼接

x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
# 最后添加主要的逻辑回归层
main_output  = Dense(1, activation='sigmoid', name='main_output')(x)
model = Model(inputs = [main_input,auxiliary_input],outputs = [main_output, auxiliary_output])
'''
现在编译模型，并给辅助损失分配一个 0.2 的权重。如果要为不同的输出指定不同的 loss_weights 或 loss，
可以使用列表或字典。 在这里，我们给 loss 参数传递单个损失函数，这个损失将用于所有的输出。
'''
# 第一种
model.compile(
	optimizer='rmsprop',
	loss='binary_crossentropy',
	loss_weights=[1., 0.2] # 顺序是 inputs = [main_input,auxiliary_input],outputs = [main_output, auxiliary_output]
)
model.fit([headline_data, additional_data], [labels, labels], epochs=50, batch_size=32)
# 第二种
model.compile(
	optimizer='rmsprop',
	loss = {'main_output':'binary_crossentropy', 'auxiliary_output':'binary_crossentropy'},
	loss_weights={'main_output':1., 'auxiliary_output':0.2}
)
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)

# 共享网络层
tweet_a = Input(shape=(280, 256))
tweet_b = Input(shape=(280, 256))

share_lstm = LSTM(64)
# 当我们重用相同的图层实例多次，图层的权重也会被重用 (它其实就是同一层)
encoded_a = share_lstm(tweet_a)
encoded_b = share_lstm(tweet_b)

merged_vector = concatenate([encoded_a, encoded_b], axis=-1)
prediction = Dense(1,activation='sigmoid')(merged_vector)

model = Model(inputs=[tweet_a, tweet_b], outputs=prediction)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, epochs=10)
