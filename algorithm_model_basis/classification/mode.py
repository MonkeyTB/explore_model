# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/4/8 11:04 
# 文件     ：mode.py
# IDE     : PyCharm

from tensorflow import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, GRU, Input, Bidirectional, Activation, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import initializers, regularizers, constraints

import pandas as pd
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from nltk import word_tokenize
import gensim
import jieba
import os

from utils.config import *
from utils.log import log


class EmbeddingVector(object):
    def __init__(self, path):
        self.model = gensim.models.Word2Vec.load(path)

    def sent2vec(self, s, type_=True):
        '''
        :param s: 句子
        :param type: 句子是否已经切词，切-True，未切-False
        :return: 句子对应的词向量
        '''
        if not type_:
            jieba.enable_parallel()  # 并行分词开启
            words = str(s).lower()
            # words = word_tokenize(words)
            words = jieba.lcut(words)
        else:
            words = s.split(' ')
        if config.stop_word:
            stwlist = [line.strip() for line in open(config.stop_path, 'r', encoding='utf-8').readlines()]
        else:
            stwlist = []

        words = [w for w in words if not w in stwlist]
        # words = [w for w in words if w.isalpha()]
        M = []
        for w in words:
            try:
                # M.append(embeddings_index[w])
                M.append(self.model[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        if type(v) != np.ndarray:
            return np.zeros(300)
        return (v / np.sqrt((v ** 2).sum()))


class BasicModels(TfidfVectorizer):
    '''
    tf-idf
    word-count
    逻辑斯底回归（Logistic Regression）
    xgboost
    '''

    def build_tokenizer(self):
        tokenize = super(BasicModels, self).build_tokenizer()
        return lambda doc: list(self.number_normalizer(tokenize(doc)))

    def number_normalizer(self, tokens):
        """ 将所有数字标记映射为一个占位符（Placeholder）。
        对于许多实际应用场景来说，以数字开头的tokens不是很有用，
        但这样tokens的存在也有一定相关性。 通过将所有数字都表示成同一个符号，可以达到降维的目的。
        """
        return ("#NUMBER" if token[0].isdigit() else token for token in tokens)

    def tf_idf(self, xtrain, xvalid):
        '''
        :return:tf-idf作为id
        '''
        if config.stop_word:
            stwlist = [line.strip() for line in open(config.stop_path, 'r', encoding='utf-8').readlines()]
        else:
            stwlist = []
        tfv = BasicModels(min_df=3,
                          max_df=0.5,
                          max_features=None,
                          ngram_range=(1, 2),
                          use_idf=True,
                          smooth_idf=True,
                          stop_words=stwlist)
        # 使用TF-IDF来fit训练集和测试集（半监督学习）
        tfv.fit(list(xtrain) + list(xvalid))
        xtrain_tfv = tfv.transform(xtrain)
        xvalid_tfv = tfv.transform(xvalid)
        return xtrain_tfv, xvalid_tfv

    def word_count(self, xtrain, xvalid):
        '''
        :return:词频计数作为id，而不是tf-idf
        '''
        if config.stop_word:
            stwlist = [line.strip() for line in open(config.stop_path, 'r', encoding='utf-8').readlines()]
        else:
            stwlist = []
        ctv = CountVectorizer(min_df=3,
                              max_df=0.5,
                              ngram_range=(1, 2),
                              stop_words=stwlist)

        # 使用 Count Vectorizer 来 fit 训练集和测试集（半监督学习）
        ctv.fit(list(xtrain) + list(xvalid))
        xtrain_ctv = ctv.transform(xtrain)
        xvalid_ctv = ctv.transform(xvalid)
        return xtrain_ctv, xvalid_ctv

    def word2vec(self, xtrain, xvalid):
        '''
        训练的词向量
        '''
        ev = EmbeddingVector(config.word2vec_model_path)
        # 对训练集和验证集使用上述函数，进行文本向量化处理
        xtrain_w2v = [ev.sent2vec(x) for x in tqdm(xtrain)]
        xvalid_w2v = [ev.sent2vec(x) for x in tqdm(xvalid)]

        xtrain_w2v = np.array(xtrain_w2v)
        xvalid_w2v = np.array(xvalid_w2v)
        return xtrain_w2v, xvalid_w2v

    def LR_train(self, x_train, y_train):
        clf = LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial')
        clf.fit(x_train, y_train)
        return clf

    def LR_test(self, clf, x_test):
        predictions = clf.predict_proba(x_test)
        return predictions

    def xgboost_train(self, x_train, y_train):
        clf = xgb.XGBClassifier(max_depth=10, n_estimators=300, colsample_bytree=0.8,
                                subsample=0.8, nthread=10, learning_rate=0.1)
        log('max_depth=10, n_estimators=300, colsample_bytree=0.8,subsample=0.8, nthread=10, learning_rate=0.1')
        if config.id_type == 'word2vec':
            clf.fit(x_train, y_train)
        else:
            clf.fit(x_train.tocsc(), y_train)  # 稀疏矩阵的表示
        return clf

    def xgboost_test(self, clf, x_test):
        if config.id_type == 'word2vec':
            predictions = clf.predict_proba(x_test)
        else:
            predictions = clf.predict_proba(x_test.tocsc())
        return predictions


class TextCnn(object):
    def __init__(self):
        pass

    def matrix_func(self, embedding_path):
        '''
        :param embedding_path: 向量文件
        :return: numpy 数组，每行一个词向量
        '''
        matrix = np.load(embedding_path)
        return matrix

    def TextCNN(self, vocab_size, seq_length, embed_size, num_classes, num_filters, filter_sizes, regularizers_lambda,
                dropout_rate):
        '''
        :param vocab_size: 词汇表大小
        :param seq_length: 句子长度
        :param embed_size: 词向量大小
        :param num_classes: num 分类
        :param num_filters: 卷积通道数
        :param filter_sizes: 卷积核大小'2,3,4'
        :param regularizers_lambda:
        :param dropout_rate:
        :return:
        '''
        inputs = Input(shape=(seq_length,), name='input_data')
        if os.path.exists(config.embedding_path):
            embddding = self.matrix_func(config.embedding_path)
        embed_initer = keras.initializers.RandomUniform(minval=-1, maxval=1)
        embed = keras.layers.Embedding(vocab_size, embed_size, embeddings_initializer=embed_initer,
                                       # weights = [embedding],
                                       input_length=seq_length,
                                       name='embdding')(inputs)
        # 单通道。如果使用真正的嵌入，你可以设置一个静态的
        embed = keras.layers.Reshape((seq_length, embed_size, 1), name='add_channel')(embed)
        pool_outputs = []
        for filter_size in list(map(int, filter_sizes.split(','))):
            filter_shape = (filter_size, embed_size)
            conv = keras.layers.Conv2D(num_filters, filter_shape,
                                       strides=(1, 1), padding='valid',
                                       data_format='channels_last',
                                       activation='relu',
                                       kernel_initializer='glorot_normal',
                                       bias_initializer=keras.initializers.constant(0.1),
                                       name='convolution_{:d}'.format(filter_size))(embed)
            max_pool_shape = (seq_length - filter_size + 1, 1)
            pool = keras.layers.MaxPooling2D(pool_size=max_pool_shape,
                                             strides=(1, 1), padding='valid',
                                             data_format='channels_last',
                                             name='max_pooling_{:d}'.format(filter_size))(conv)
            pool_outputs.append(pool)
        pool_outputs = keras.layers.concatenate(pool_outputs, axis=-1, name='concatenate')
        pool_outputs = keras.layers.Flatten(data_format='channels_last', name='flatten')(pool_outputs)
        pool_outputs = keras.layers.Dropout(dropout_rate, name='dropout')(pool_outputs)

        outputs = keras.layers.Dense(num_classes, activation='sigmoid',
                                     kernel_initializer='glorot_normal',
                                     bias_initializer=keras.initializers.constant(0.1),
                                     kernel_regularizer=keras.regularizers.l2(regularizers_lambda),
                                     bias_regularizer=keras.regularizers.l2(regularizers_lambda),
                                     name='dense')(pool_outputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def train(self, x_train, y_train, vocab_size, seq_length, save_path, timestamp):
        log('\n Train ...')
        log(vocab_size)
        model = self.TextCNN(vocab_size, seq_length, config.embedding_size, config.numclass, config.num_filters,
                             config.kernel_size, config.regularizers_lambda, config.dropout)
        model.summary()
        model.compile(tf.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        tb_callback = keras.callbacks.TensorBoard(histogram_freq=0.1, write_graph=True,
                                                  write_grads=True, write_images=True,
                                                  embeddings_freq=0.5, update_freq='batch')
        history = model.fit(x=x_train, y=y_train, batch_size=config.batch_size, epochs=config.epochs,
                            callbacks=[tb_callback], validation_split=0.1, shuffle=True)
        log("\nSaving model...")
        keras.models.save_model(model, save_path)
        log(history.history)

    def test(self, model, x_test, y_test):
        log("Test...")
        y_pred_one_hot = model.predict(x=x_test, batch_size=1, verbose=1)
        # y_pred = tf.math.argmax(y_pred_one_hot, axis=1)
        return y_pred_one_hot


class TextCnnNew(object):
    def __init__(self):
        pass

    def matrix_func(self, embedding_path):
        '''
        :param embedding_path: 向量文件
        :return: numpy 数组，每行一个词向量
        '''
        matrix = np.load(embedding_path)
        return matrix

    def TextCnnNew(self, vocab_size, seq_length, embed_size, num_classes, num_filters, filter_sizes,
                   regularizers_lambda, dropout_rate):
        '''
        :param vocab_size: 词汇表大小
        :param seq_length: 句子长度
        :param embed_size: 词向量大小
        :param num_classes: num 分类
        :param num_filters: 卷积通道数
        :param filter_sizes: 卷积核大小'2,3,4'
        :param regularizers_lambda:
        :param dropout_rate:
        :return:
        '''
        inputs = Input(shape=(seq_length,), name='input_data')
        if os.path.exists(config.embedding_path):
            embddding = self.matrix_func(config.embedding_path)
        embed_initer = keras.initializers.RandomUniform(minval=-1, maxval=1)
        embed1 = keras.layers.Embedding(vocab_size, 100, embeddings_initializer=embed_initer,
                                        # weights = [embedding],
                                        input_length=seq_length,
                                        name='embdding1')(inputs)
        embed2 = keras.layers.Embedding(vocab_size, 200, embeddings_initializer=embed_initer,
                                        # weights = [embedding],
                                        input_length=seq_length,
                                        name='embdding2')(inputs)
        embed3 = keras.layers.Embedding(vocab_size, 300, embeddings_initializer=embed_initer,
                                        # weights = [embedding],
                                        input_length=seq_length,
                                        name='embdding3')(inputs)

        embed1 = keras.layers.Dense(50, name='dense1')(embed1)
        embed2 = keras.layers.Dense(50, name='dense2')(embed2)
        embed3 = keras.layers.Dense(50, name='dense3')(embed3)

        embed1 = keras.layers.BatchNormalization()(embed1)
        embed2 = keras.layers.BatchNormalization()(embed2)
        embed3 = keras.layers.BatchNormalization()(embed3)

        embed = keras.layers.concatenate([embed1, embed2, embed3], axis=-1, name='concatenate1')
        embed = keras.layers.Reshape((seq_length, 150, 1), name='add_channel')(embed)
        #         embed = keras.layers.Flatten(data_format='channels_last',name='flatten1')(embed)
        pool_outputs = []
        for filter_size in list(map(int, filter_sizes.split(','))):
            filter_shape = (filter_size, 150)
            conv = keras.layers.Conv2D(num_filters, filter_shape,
                                       strides=(1, 1), padding='valid',
                                       data_format='channels_last',
                                       activation='relu',
                                       kernel_initializer='glorot_normal',
                                       bias_initializer=keras.initializers.constant(0.1),
                                       name='convolution_{:d}'.format(filter_size))(embed)
            max_pool_shape = (seq_length - filter_size + 1, 1)
            pool = keras.layers.MaxPooling2D(pool_size=max_pool_shape,
                                             strides=(1, 1), padding='valid',
                                             data_format='channels_last',
                                             name='max_pooling_{:d}'.format(filter_size))(conv)
            pool_outputs.append(pool)
        pool_outputs = keras.layers.concatenate(pool_outputs, axis=-1, name='concatenate')
        pool_outputs = keras.layers.Flatten(data_format='channels_last', name='flatten')(pool_outputs)
        pool_outputs = keras.layers.Dropout(dropout_rate, name='dropout')(pool_outputs)

        outputs = keras.layers.Dense(num_classes, activation='softmax',
                                     kernel_initializer='glorot_normal',
                                     bias_initializer=keras.initializers.constant(0.1),
                                     kernel_regularizer=keras.regularizers.l2(regularizers_lambda),
                                     bias_regularizer=keras.regularizers.l2(regularizers_lambda),
                                     name='dense')(pool_outputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def train(self, x_train, y_train, vocab_size, seq_length, save_path, timestamp):
        log('\n Train ...')
        log(vocab_size)
        model = self.TextCnnNew(vocab_size, seq_length, config.embedding_size, config.numclass, config.num_filters,
                                config.kernel_size, config.regularizers_lambda, config.dropout)
        model.summary()
        model.compile(tf.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        tb_callback = keras.callbacks.TensorBoard(histogram_freq=0.1, write_graph=True,
                                                  write_grads=True, write_images=True,
                                                  embeddings_freq=0.5, update_freq='batch')
        history = model.fit(x=x_train, y=y_train, batch_size=config.batch_size, epochs=config.epochs,
                            callbacks=[tb_callback], validation_split=0.1, shuffle=True)
        log("\nSaving model...")
        keras.models.save_model(model, save_path)
        log(history.history)

    def test(self, model, x_test, y_test):
        log("Test...")
        y_pred_one_hot = model.predict(x=x_test, batch_size=1, verbose=1)
        # y_pred = tf.math.argmax(y_pred_one_hot, axis=1)
        return y_pred_one_hot


class TextCnnMultiDim(object):
    def __init__(self):
        pass

    def matrix_func(self, embedding_path):
        '''
        :param embedding_path: 向量文件
        :return: numpy 数组，每行一个词向量
        '''
        matrix = np.load(embedding_path)
        return matrix

    def TextCnnMultiDim(self, char_size, word_size, cx_size, seq_length, embed_size, num_classes, num_filters,
                        filter_sizes,
                        regularizers_lambda, dropout_rate):
        '''
        :param vocab_size: 词汇表大小
        :param seq_length: 句子长度
        :param embed_size: 词向量大小
        :param num_classes: num 分类
        :param num_filters: 卷积通道数
        :param filter_sizes: 卷积核大小'2,3,4'
        :param regularizers_lambda:
        :param dropout_rate:
        :return:
        '''
        inputs1 = Input(shape=(seq_length,), name='input_data1')
        inputs2 = Input(shape=(seq_length,), name='input_data2')
        inputs3 = Input(shape=(seq_length,), name='input_data3')
        if os.path.exists(config.embedding_path):
            embddding = self.matrix_func(config.embedding_path)
        embed_initer = keras.initializers.RandomUniform(minval=-1, maxval=1)
        embed1 = keras.layers.Embedding(char_size, config.embedding_size, embeddings_initializer=embed_initer,
                                        # weights = [embedding],
                                        input_length=seq_length,
                                        name='embdding1')(inputs1)
        embed2 = keras.layers.Embedding(word_size, config.embedding_size, embeddings_initializer=embed_initer,
                                        # weights = [embedding],
                                        input_length=seq_length,
                                        name='embdding2')(inputs2)
        embed3 = keras.layers.Embedding(cx_size, config.embedding_size, embeddings_initializer=embed_initer,
                                        # weights = [embedding],
                                        input_length=seq_length,
                                        name='embdding3')(inputs3)

        embed1 = keras.layers.Reshape((seq_length, config.embedding_size, 1), name='embed1')(embed1)
        embed2 = keras.layers.Reshape((seq_length, config.embedding_size, 1), name='embed2')(embed2)
        embed3 = keras.layers.Reshape((seq_length, config.embedding_size, 1), name='embed3')(embed3)
        pool_outputs = []
        embed = [embed1, embed2, embed3]
        for i in range(len(embed)):
            for filter_size in list(map(int, filter_sizes.split(','))):
                filter_shape = (filter_size, config.embedding_size)
                conv = keras.layers.Conv2D(num_filters, filter_shape,
                                           strides=(1, 1), padding='valid',
                                           data_format='channels_last',
                                           activation='relu',
                                           kernel_initializer='glorot_normal',
                                           bias_initializer=keras.initializers.constant(0.1),
                                           name='convolution_{:d}_{:d}'.format(filter_size, i))(embed[i])
                max_pool_shape = (seq_length - filter_size + 1, 1)
                pool = keras.layers.MaxPooling2D(pool_size=max_pool_shape,
                                                 strides=(1, 1), padding='valid',
                                                 data_format='channels_last',
                                                 name='max_pooling_{:d}_{:d}'.format(filter_size, i))(conv)
                pool_outputs.append(pool)
        pool_outputs = keras.layers.concatenate(pool_outputs, axis=-1, name='concatenate')
        pool_outputs = keras.layers.Flatten(data_format='channels_last', name='flatten')(pool_outputs)
        pool_outputs = keras.layers.Dropout(dropout_rate, name='dropout')(pool_outputs)

        outputs = keras.layers.Dense(num_classes, activation='softmax',
                                     kernel_initializer='glorot_normal',
                                     bias_initializer=keras.initializers.constant(0.1),
                                     kernel_regularizer=keras.regularizers.l2(regularizers_lambda),
                                     bias_regularizer=keras.regularizers.l2(regularizers_lambda),
                                     name='dense')(pool_outputs)
        model = keras.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        return model

    def train(self, xtrain_char, xtrain_word, xtrain_cx, y_train, char_size, word_size, cx_size, seq_length, save_path,
              timestamp):
        log('\n Train ...')
        model = self.TextCnnMultiDim(char_size, word_size, cx_size, seq_length, config.embedding_size, config.numclass,
                                     config.num_filters,
                                     config.kernel_size, config.regularizers_lambda, config.dropout)
        model.summary()
        model.compile(tf.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        tb_callback = keras.callbacks.TensorBoard(histogram_freq=0.1, write_graph=True,
                                                  write_grads=True, write_images=True,
                                                  embeddings_freq=0.5, update_freq='batch')
        history = model.fit(x=[xtrain_char, xtrain_word, xtrain_cx], y=y_train, batch_size=config.batch_size,
                            epochs=config.epochs,
                            callbacks=[tb_callback], validation_split=0.1, shuffle=True)
        log("\nSaving model...")
        keras.models.save_model(model, save_path)
        log(history.history)

    def test(self, model, xtest_char, xtest_word, xtest_cx, y_test):
        log("Test...")
        y_pred_one_hot = model.predict(x=[xtest_char, xtest_word, xtest_cx], batch_size=1, verbose=1)
        # y_pred = tf.math.argmax(y_pred_one_hot, axis=1)
        return y_pred_one_hot


class TextRCNN(object):
    def __init__(self):
        pass

    def matrix_func(self, embedding_path):
        '''
        :param embedding_path: 向量文件
        :return: numpy 数组，每行一个词向量
        '''
        matrix = np.load(embedding_path)
        return matrix

    def RCNN(self, vocab_size, seq_length, embed_size, num_classes,
             filter_sizes, regularizers_lambda, dropout_rate):
        '''
        :param vocab_size: 词汇表大小
        :param seq_length: 句子长度
        :param embed_size: 词向量大小
        :param num_classes: num 分类
        :param num_filters: 卷积通道数
        :param filter_sizes: 卷积核大小'2,3,4'
        :param regularizers_lambda:
        :param dropout_rate:
        :return:
        '''
        input_current = Input(shape=(seq_length,), name='input_current')
        input_left = Input(shape=(seq_length,), name='input_left')
        input_right = Input(shape=(seq_length,), name='input_right')
        if os.path.exists(config.embedding_path):
            embedding = self.matrix_func(config.embedding_path)
        embed_initer = keras.initializers.RandomUniform(minval=-1, maxval=1)
        embed_current = keras.layers.Embedding(vocab_size, config.embedding_size, embeddings_initializer=embed_initer,
                                               # weights = [embedding],
                                               input_length=seq_length,
                                               name='embdding_current')(input_current)
        embed_left = keras.layers.Embedding(vocab_size, config.embedding_size, embeddings_initializer=embed_initer,
                                            # weights = [embedding],
                                            input_length=seq_length,
                                            name='embdding_left')(input_left)
        embed_right = keras.layers.Embedding(vocab_size, config.embedding_size, embeddings_initializer=embed_initer,
                                             # weights = [embedding],
                                             input_length=seq_length,
                                             name='embdding_right')(input_right)

        #         x_left = keras.layers.SimpleRNN(128,return_sequences=True,name='rnn_left')(embed_left) # seq_lenght * 128
        #         x_right = keras.layers.SimpleRNN(128,return_sequences=True,go_backwards=True,name='rnn_right')(embed_right) # seq_lenght * 128
        x_left = keras.layers.GRU(128, return_sequences=True, name='rnn_left')(embed_left)  # seq_lenght * 128
        x_right = keras.layers.GRU(128, return_sequences=True, go_backwards=True, name='rnn_right')(
            embed_right)  # seq_lenght * 128
        x_right = keras.layers.Lambda(lambda x: K.reverse(x, axes=1), name='lambda')(x_right)

        x = keras.layers.concatenate([x_left, embed_current, x_right], axis=2,
                                     name='concat')  # 128 + embedding_size + 128

        x = keras.layers.Conv1D(64, kernel_size=1, activation='tanh', name='conv1d')(x)
        x = keras.layers.GlobalMaxPooling1D(name='global_maxpooling')(x)
        outputs = keras.layers.Dense(num_classes, activation='softmax',
                                     kernel_initializer='glorot_normal',
                                     bias_initializer=keras.initializers.constant(0.1),
                                     kernel_regularizer=keras.regularizers.l2(regularizers_lambda),
                                     bias_regularizer=keras.regularizers.l2(regularizers_lambda),
                                     name='dense')(x)
        model = keras.Model(inputs=[input_current, input_left, input_right], outputs=outputs)
        return model

    def train(self, xtrain_current, xtrain_left, xtrain_right, y_train, vocab_size, seq_length, save_path, timestamp):
        log('\n Train ...')
        model = self.RCNN(vocab_size, seq_length, config.embedding_size, config.numclass,
                          config.kernel_size, config.regularizers_lambda, config.dropout)
        model.summary()
        model.compile(tf.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        tb_callback = keras.callbacks.TensorBoard(histogram_freq=0.1, write_graph=True,
                                                  write_grads=True, write_images=True,
                                                  embeddings_freq=0.5, update_freq='batch')
        history = model.fit(x=[xtrain_current, xtrain_left, xtrain_right], y=y_train, batch_size=config.batch_size,
                            epochs=config.epochs,
                            callbacks=[tb_callback], validation_split=0.1, shuffle=True)
        log("\nSaving model...")
        keras.models.save_model(model, save_path)
        log(history.history)

    def test(self, model, xtrain_current, xtrain_left, xtrain_right, y_test):
        log("Test...")
        y_pred_one_hot = model.predict(x=[xtrain_current, xtrain_left, xtrain_right], batch_size=1, verbose=1)
        # y_pred = tf.math.argmax(y_pred_one_hot, axis=1)
        return y_pred_one_hot


class TextRNN(object):
    def __init__(self):
        pass

    def matrix_func(self, embedding_path):
        '''
        :param embedding_path: 向量文件
        :return: numpy 数组，每行一个词向量
        '''
        matrix = np.load(embedding_path)
        return matrix

    def RNN(self, vocab_size, seq_length, embed_size, num_classes, regularizers_lambda):
        '''
        :param vocab_size: 词汇表大小
        :param seq_length: 句子长度
        :param embed_size: 词向量大小
        :param num_classes: num 分类
        :param regularizers_lambda:
        :param dropout_rate:
        :return:
        '''
        input = Input(shape=(seq_length,), name='input')
        if os.path.exists(config.embedding_path):
            embedding = self.matrix_func(config.embedding_path)
        embed_initer = keras.initializers.RandomUniform(minval=-1, maxval=1)
        embed = keras.layers.Embedding(vocab_size, config.embedding_size, embeddings_initializer=embed_initer,
                                       # weights = [embedding],
                                       input_length=seq_length,
                                       name='embdding')(input)

        x = keras.layers.Bidirectional(GRU(128, return_sequences=False, name='gru'))(embed)
        #         x = keras.layers.Flatten(data_format='channels_last', name='flatten')(x)
        outputs = keras.layers.Dense(num_classes, activation='softmax',
                                     kernel_initializer='glorot_normal',
                                     bias_initializer=keras.initializers.constant(0.1),
                                     kernel_regularizer=keras.regularizers.l2(regularizers_lambda),
                                     bias_regularizer=keras.regularizers.l2(regularizers_lambda),
                                     name='dense')(x)
        model = keras.Model(inputs=input, outputs=outputs)
        return model

    def train(self, xtrain, y_train, vocab_size, seq_length, save_path, timestamp):
        log('\n Train ...')
        model = self.RNN(vocab_size, seq_length, config.embedding_size, config.numclass, config.regularizers_lambda, )
        model.summary()
        model.compile(tf.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        tb_callback = keras.callbacks.TensorBoard(histogram_freq=0.1, write_graph=True,
                                                  write_grads=True, write_images=True,
                                                  embeddings_freq=0.5, update_freq='batch')
        history = model.fit(x=xtrain, y=y_train, batch_size=config.batch_size,
                            epochs=config.epochs, callbacks=[tb_callback], validation_split=0.1, shuffle=True)
        log("\nSaving model...")
        keras.models.save_model(model, save_path)
        log(history.history)

    def test(self, model, xtest, y_test):
        log("Test...")
        y_pred_one_hot = model.predict(x=xtest, batch_size=1, verbose=1)
        return y_pred_one_hot


class Attention(Layer):
    '''
    实现实时数据的注意力机制
    Input：3D tensor (samples, steps, features) (batch_size, seq_len, units)
    output: 2D tensor (samples, features) (batch_size, units)
    kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
      example:
        # 1
        model.add(LSTM(64, return_sequence=True)) # 每个时刻的输出，计算attention
        model.add(Attention())
        - next add a Dense layer (for classification/regression) or whatever
        # 2
        hidden = LSTM(64, return_sequence=True)(words)
        sententce = Attention()(hidden)
        - next add a Dense layer (for classification/regression) or whatever
    '''

    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_w'.format(self.name), )
        #                                 regularizer=self.W_regularizer,
        #                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name), )
        #                                     regularizer=self.b_regularizer,
        #                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.build = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, -1))),
                      (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)
        a = K.exp(e)

        # apply mask after the exp. will be re-normalized(重新规范化) next
        if mask is not None:
            a *= K.cast(mask, K.floatx())  # K.floatx() : 'float32'
        # 避免前期结果为 nan，加一个很小的数 K.epsilon()
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


class TextAttBiRNN(Attention):
    def __init__(self):
        pass

    #         self.att = Attention(step_dim=seq_length)

    def matrix_func(self, embedding_path):
        '''
        :param embedding_path: 向量文件
        :return: numpy 数组，每行一个词向量
        '''
        matrix = np.load(embedding_path)
        return matrix

    def AttBiRNN(self, vocab_size, seq_length, embed_size, num_classes, regularizers_lambda):
        '''
        :param vocab_size: 词汇表大小
        :param seq_length: 句子长度
        :param embed_size: 词向量大小
        :param num_classes: num 分类
        :param regularizers_lambda:
        :param dropout_rate:
        :return:
        '''
        input = Input(shape=(seq_length,), name='input')
        if os.path.exists(config.embedding_path):
            embedding = self.matrix_func(config.embedding_path)
        embed_initer = keras.initializers.RandomUniform(minval=-1, maxval=1)
        embed = keras.layers.Embedding(vocab_size, config.embedding_size, embeddings_initializer=embed_initer,
                                       # weights = [embedding],
                                       input_length=seq_length,
                                       name='embdding')(input)
        #         print('embed shape:',embed.shape) # (None, 81, 300)
        x = keras.layers.Bidirectional(GRU(128, return_sequences=True, name='gru'))(embed)
        #         print('BiGRU shape:',x.shape) # (None, 81, 256)
        x = Attention(seq_length)(x)
        outputs = keras.layers.Dense(num_classes, activation='softmax',
                                     kernel_initializer='glorot_normal',
                                     bias_initializer=keras.initializers.constant(0.1),
                                     kernel_regularizer=keras.regularizers.l2(regularizers_lambda),
                                     bias_regularizer=keras.regularizers.l2(regularizers_lambda),
                                     name='dense')(x)
        model = keras.Model(inputs=input, outputs=outputs)
        return model

    def train(self, xtrain, y_train, vocab_size, seq_length, save_path, timestamp):
        log('\n Train ...')
        model = self.AttBiRNN(vocab_size, seq_length, config.embedding_size, config.numclass,
                              config.regularizers_lambda, )
        model.summary()
        model.compile(tf.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        tb_callback = keras.callbacks.TensorBoard(histogram_freq=0.1, write_graph=True,
                                                  write_grads=True, write_images=True,
                                                  embeddings_freq=0.5, update_freq='batch')
        history = model.fit(x=xtrain, y=y_train, batch_size=config.batch_size,
                            epochs=config.epochs, callbacks=[tb_callback], validation_split=0.1, shuffle=True)
        log("\nSaving model...")
        keras.models.save_model(model, save_path)
        log(history.history)

    def test(self, model, xtest, y_test):
        log("Test...")
        y_pred_one_hot = model.predict(x=xtest, batch_size=1, verbose=1)
        return y_pred_one_hot

