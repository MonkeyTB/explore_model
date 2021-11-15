# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/11/3 10:31 
# 文件     ：train.py
# IDE     : PyCharm

import tensorflow.keras as keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.layers import Embedding, Input, GlobalAvgPool1D, Dropout, Dense
from Attention_keras import Position_Embedding,MultiHeadAttention
from matplotlib import pyplot as plt
import pandas as pd

max_features = 20000
print('Loading data ...')
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)

# one-hot
y_train, y_test = pd.get_dummies(y_train), pd.get_dummies(y_test)
print(len(x_train), 'train sequence')
print(len(x_test), 'test sequence')

# 数据归一化处理
maxlen = 64
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print(x_train.shape, 'train pad sequence')
print(x_test.shape, 'test pad sequence')

batch_size = 128
input = Input(shape=(None,))

embedding = Embedding(max_features, 128)(input) # (None, None, 128)
embedding = Position_Embedding()(embedding) # (None, None, 128)

O_seq = MultiHeadAttention(8,26)([embedding, embedding, embedding]) # (None, None, 128)
O_seq = GlobalAvgPool1D()(O_seq) # (None, 128)
O_seq = Dropout(0.5)(O_seq)

output = Dense(2,activation='softmax')(O_seq) # (None, 2)

model = Model(inputs=input, outputs=output)
opt = Adam(lr=0.0005)
loss = 'categorical_crossentropy'
model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
print(model.summary())

print('train ...')
model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test,y_test))
model.save('imbd.h5')