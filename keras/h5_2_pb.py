# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/12/7 14:40 
# 文件     ：h5_2_pb.py
# IDE     : PyCharm]
'''
h5模型重载为pd文件，各模型尝试
'''
import os
os.environ['TF_KERAS'] = '1'
import numpy as np
import pandas as pd
from bert4keras.backend import keras,K
from bert4keras.layers import Loss, Embedding
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model, BERT
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import Lambda, Dense
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
############################## 1、Bert_Ptuning model ######################################
# class PtuningEmbedding(Embedding):
#     '''
#     定义新的embedding层，只优化部分token
#     '''
#     def call(self, inputs, mode='embedding'):
#         embeddings = self.embeddings
#         embeddings_sg = K.stop_gradient(embeddings)
#         mask = np.zeros((K.int_shape(embeddings)[0],1))
#         mask[1:9] += 1
#         self.embeddings = embeddings * mask + embeddings_sg * (1-mask)
#         outputs = super(PtuningEmbedding, self).call(inputs, mode)
#         self.embeddings = embeddings
#         return outputs
# class PtuningBERT(BERT):
#     '''
#     替换原来的embedding
#     '''
#     def apply(self, inputs=None, layer=None, arguments=None, **kwargs):
#         if layer is Embedding:
#             layer = PtuningEmbedding
#         return super(PtuningBERT, self).apply(inputs, layer, arguments, **kwargs)
# model = 'model/Bert_Ptuning.h5'
# base = 'model/pb/'
# keras_model = load_model(model,compile=False,custom_objects={'PtuningEmbedding': PtuningEmbedding})
# keras_model.save(base + '/Bert_Ptuning/1',save_format='tf') # <====注意model path里面的1是代表版本号，必须有这个不然tf serving 会报找不到可以serve的model


############################## 2、RE_bert model ######################################
# from bert4keras.backend import keras, K, batch_gather
# class TotalLoss(Loss):
#     """subject_loss与object_loss之和，都是二分类交叉熵
#     """
#     def compute_loss(self, inputs, mask=None):
#         subject_labels, object_labels = inputs[:2]
#         subject_preds, object_preds, _ = inputs[2:]
#         if mask[4] is None:
#             mask = 1.0
#         else:
#             mask = K.cast(mask[4], K.floatx())
#         # sujuect部分loss
#         subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
#         subject_loss = K.mean(subject_loss, 2)
#         subject_loss = K.sum(subject_loss * mask) / K.sum(mask)
#         # object部分loss
#         object_loss = K.binary_crossentropy(object_labels, object_preds)
#         object_loss = K.sum(K.mean(object_loss, 3), 2)
#         object_loss = K.sum(object_loss * mask) / K.sum(mask)
#         # 总的loss
#         return subject_loss + object_loss
# model = 'model/RE_bert.h5'
# base = 'model/pb/'
# keras_model = load_model(model, compile = False, custom_objects = {'TotalLoss' : TotalLoss, 'batch_gather' : batch_gather})
# keras_model.save(base + '/RE_bert/1',save_format='tf') # <====注意model path里面的1是代表版本号，必须有这个不然tf serving 会报找不到可以serve的model
############################# 3、GlobalPoint #############################################
model = 'model/GlobalPoint.h5'
base = 'model/pb/'
keras_model = load_model(model, compile = False)
keras_model.save(base + '/GlobalPoint/1',save_format='tf') # <====注意model path里面的1是代表版本号，必须有这个不然tf serving 会报找不到可以serve的model