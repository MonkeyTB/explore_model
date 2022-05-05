# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2022/3/17 14:19
# 文件     ：bert_theseus.py
# IDE     : PyCharm
# 论文：https://arxiv.org/abs/2002.02925

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Input, Lambda, Dense, Layer
from keras.models import Model
import pandas as pd

num_classes = 5
maxlen = 128
batch_size = 32

# BERT base
config_path = '../pre_mode/bert-base-cased-ch/bert_config.json'
checkpoint_path = '../pre_mode/bert-base-cased-ch/bert_model.ckpt'
dict_path = '../pre_mode/bert-base-cased-ch/vocab.txt'


def load_data(filename,rate):
    dict_lable = {'其他口号': 0, '工作时间': 0, '要求': 2, '描述': 3, '薪资福利': 4, '公司介绍': 0, '工作地址': 0, '岗位晋升/职业发展': 4, '联系方式': 0,'职位名称':1}
    T, D = [], []
    df = pd.read_excel(filename).sample(frac=1)
    df_train = df[:int(df.shape[0]*rate)]
    df_val = df[int(df.shape[0]*rate):]
    for key,row in df_train.iterrows():
        T.append((str(row.content), dict_lable[row.label]))
    for key,row in df_val.iterrows():
        D.append((str(row.content), dict_lable[row.label]))
    return T,D


# 加载数据集
train_data, valid_data = load_data(
    'Bert_Theseus_data/corpus.xlsx', 0.9
)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


class BinaryRandomChoice(Layer):
    """随机二选一
    """
    def __init__(self, **kwargs):
        super(BinaryRandomChoice, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[1]

    def call(self, inputs):
        source, target = inputs
        mask = K.random_binomial(shape=[1], p=0.5) # 产生一个二项分布
        '''
        :shape : 一个整数元组，需要创建张量的尺寸
        :p : 二项分布的概率
        '''
        output = mask * source + (1 - mask) * target
        return K.in_train_phase(output, target) # 训练阶段选择output，其他阶段选择target

    def compute_output_shape(self, input_shape):
        return input_shape[1]


def bert_of_theseus(predecessor, successor, classfier):
    """bert of theseus
    : predesessor: 前辈
    : successor: 传承者
    : classfier: 分类层?/
    """
    inputs = predecessor.inputs
    # 固定住已经训练好的层
    for layer in predecessor.model.layers:
        layer.trainable = False
    classfier.trainable = False
    # Embedding层替换
    predecessor_outputs = predecessor.apply_embeddings(inputs) # inputs's embedding add
    successor_outputs = successor.apply_embeddings(inputs)
    outputs = BinaryRandomChoice()([predecessor_outputs, successor_outputs])
    # Transformer层替换
    layers_per_module = predecessor.num_hidden_layers // successor.num_hidden_layers
    for index in range(successor.num_hidden_layers): # successor 的每一层
        predecessor_outputs = outputs
        for sub_index in range(layers_per_module): # predecessor 的layers_per_module层合并为一层
            predecessor_outputs = predecessor.apply_main_layers(
                predecessor_outputs, layers_per_module * index + sub_index
            )
        successor_outputs = successor.apply_main_layers(outputs, index)
        outputs = BinaryRandomChoice()([predecessor_outputs, successor_outputs])
    # 返回模型
    outputs = classfier(outputs)
    model = Model(inputs, outputs)
    return model


def evaluate(data, model):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
#         print('y_pred', y_pred , 'y_true', y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self, savename):
        self.best_val_acc = 0.
        self.savename = savename

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator, self.model)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights(self.savename)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )
class Evaluator_S(keras.callbacks.Callback):
    def __init__(self, savename):
        self.best_val_acc = 0.
        self.savename = savename

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator, self.model)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save(self.savename)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )
# 加载预训练模型（12层）
predecessor = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
    prefix='Predecessor-'
)

# 加载预训练模型（3层）
successor = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
    num_hidden_layers=3,
    prefix='Successor-'
)

# 判别模型
x_in = Input(shape=K.int_shape(predecessor.output)[1:])
x = Lambda(lambda x: x[:, 0])(x_in)
x = Dense(units=num_classes, activation='softmax')(x)
classfier = Model(x_in, x)

predecessor_model = Model(predecessor.inputs, classfier(predecessor.output))
predecessor_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)
predecessor_model.summary()

successor_model = Model(successor.inputs, classfier(successor.output))
successor_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)
successor_model.summary()

theseus_model = bert_of_theseus(predecessor, successor, classfier)
theseus_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)
theseus_model.summary()

if __name__ == '__main__':

    # 训练predecessor
    predecessor_evaluator = Evaluator('Bert_Theseus_data/best_predecessor.weights')
    predecessor_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=5,
        callbacks=[predecessor_evaluator]
    )

    # 训练theseus
    theseus_evaluator = Evaluator('Bert_Theseus_data/best_theseus.weights')
    theseus_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[theseus_evaluator]
    )
    theseus_model.load_weights('Bert_Theseus_data/best_theseus.weights')

    # 训练successor
    successor_evaluator = Evaluator_S('model/BertTheseusSuccessor.h5')
    successor_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=5,
        callbacks=[successor_evaluator]
    )