# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/11/23 10:44 
# 文件     ：kg_with_ds.py
# IDE     : PyCharm

import os
# os.environ['TF_KERAS'] = '1'
os.environ['KERAS_BACKEND']='tensorflow'
import json
import numpy as np
from random import choice
from tqdm import tqdm
import pyhanlp
from gensim.models import Word2Vec
import re, os


mode = 0
char_size = 128
maxlen = 512


word2vec = Word2Vec.load('word2vec_baike/word2vec_baike')


id2word = {i+1:j for i,j in enumerate(word2vec.wv.index_to_key)} # 百科词向量对应的词表 id2word  index2word
word2id = {j:i for i,j in id2word.items()}
word2vec = word2vec.syn1neg # word 对应的 embedding (1056283, 256)
word_size = word2vec.shape[1] # embedding size
word2vec = np.concatenate([np.zeros((1, word_size)), word2vec]) # 第0行，padding向量


def tokenize(s):
    '''
    切词函数, HanLP
    '''
    return [i.word for i in pyhanlp.HanLP.segment(s)]


def sent2vec(S):
    """S格式：[[w1, w2]]
    """
    V = []
    for s in S: # s = [w1, w2]
        V.append([])
        for w in s: # w1, w2, ...
            for _ in w:
                V[-1].append(word2id.get(w, 0))
    V = seq_padding(V)
    V = word2vec[V]
    return V


total_data = json.load(open('data/train_data_me.json', encoding='utf-8'))
id2predicate, predicate2id = json.load(open('data/all_50_schemas_me.json', encoding='utf-8')) # 关系标签
id2predicate = {int(i):j for i,j in id2predicate.items()}
id2char, char2id = json.load(open('data/all_chars_me.json', encoding='utf-8')) # 字 id 关系
num_classes = len(id2predicate) # 分类数量


# 随机打乱数据
if not os.path.exists('data/random_order_vote.json'):
    random_order = list( range(len(total_data)) )
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('data/random_order_vote.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('data/random_order_vote.json'))

# 打乱数据切分训练和验证
train_data = [total_data[j] for i, j in enumerate(random_order) if i % 8 != mode][:1000]
dev_data = [total_data[j] for i, j in enumerate(random_order) if i % 8 == mode][:100]


predicates = {} # 格式：{predicate: [(subject, predicate, object)]}


def repair(d):
    d['text'] = d['text'].lower()
    something = re.findall(u'《([^《》]*?)》', d['text'])
    something = [s.strip() for s in something]
    zhuanji = []
    gequ = []
    for sp in d['spo_list']:
        sp[0] = sp[0].strip(u'《》').strip().lower()
        sp[2] = sp[2].strip(u'《》').strip().lower()
        for some in something:
            if sp[0] in some and d['text'].count(sp[0]) == 1:
                sp[0] = some
        if sp[1] == u'所属专辑':
            zhuanji.append(sp[2])
            gequ.append(sp[0])
    spo_list = []
    for sp in d['spo_list']:
        if sp[1] in [u'歌手', u'作词', u'作曲']:
            if sp[0] in zhuanji and sp[0] not in gequ:
                continue
        spo_list.append(tuple(sp))
    d['spo_list'] = spo_list


for d in train_data:
    repair(d)
    for sp in d['spo_list']:
        if sp[1] not in predicates:
            predicates[sp[1]] = []
        predicates[sp[1]].append(sp) # 按关系分类，保存了数据 {predicate: [(subject, predicate, object)]}


for d in dev_data:
    repair(d)


def random_generate(d, spo_list_key):
    '''
    整体可以理解为对同关系下的做了一个数据增强，提高模型的鲁棒性
    '''
    r = np.random.random() # 随机生成0-1
    if r > 0.5:
        return d
    else:
        k = np.random.randint(len(d[spo_list_key])) # 随机抽取第k组关系
        spi = d[spo_list_key][k]
        k = np.random.randint(len(predicates[spi[1]])) # 从predicates中随机抽第k组关系下的某一对
        spo = predicates[spi[1]][k] # 拿出这一对
        F = lambda s: s.replace(spi[0], spo[0]).replace(spi[2], spo[2]) # 拿出的和当前的做个替换
        text = F(d['text'])
        spo_list = [(F(sp[0]), sp[1], F(sp[2])) for sp in d[spo_list_key]]
        return {'text': text, spo_list_key: spo_list}


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list( range(len(self.data)) )
            np.random.shuffle(idxs)
            T1, T2, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], [], []
            for i in idxs:
                spo_list_key = 'spo_list' # if np.random.random() > 0.5 else 'spo_list_with_pred'
                d = random_generate(self.data[i], spo_list_key)
                text = d['text'][:maxlen]
                text_words = tokenize(text) # HanLP 切词
                text = ''.join(text_words)
                items = {} # 构造成指针 {(sp_start,sp_end):(ob_start,ob-end,p_id)}
                for sp in d[spo_list_key]:
                    subjectid = text.find(sp[0]) # 主实体起始 id，代码中用find的方式，实际工程中可以直接标注位置，从而直接获得位置，find的方式可能会存在多词多位置问题
                    objectid = text.find(sp[2])  # 客实体起始 id
                    if subjectid != -1 and objectid != -1:
                        key = (subjectid, subjectid+len(sp[0])) # 主实体(start_id, end_id)
                        if key not in items:
                            items[key] = []
                        items[key].append((objectid,
                                           objectid+len(sp[2]),
                                           predicate2id[sp[1]]))
                if items: # {(sp_start,sp_end):(ob_start,ob-end,p_id)}
                    '''
                    T1:字
                    T2:词
                    S1:主实体起始标签
                    S2:主实体结束标签
                    K1:送入模型的S的起始位置
                    K2:送入模型的S的结束位置
                    O1:客实体起始标签
                    O2:客实体结束标签
                    '''
                    T1.append([char2id.get(c, 1) for c in text]) # 1 是unk，0 是padding  字
                    T2.append(text_words) # 词
                    s1, s2 = np.zeros(len(text)), np.zeros(len(text)) # 主实体起始位置标记为 1
                    for j in items:
                        s1[j[0]] = 1
                        s2[j[1]-1] = 1
                    k1, k2 = np.array( list(items.keys()) ).T
                    # array([[1, 6],[2, 7]]),转置了,[1,6]为两个实体的开始位置
                    k1 = choice(k1) # 选择一个开始位置
                    k2 = choice(k2[k2 >= k1]) # 选择开始位置对应的结束位置,可能会取到不是一个实体边界
                    '''
                    这样做是出于这样的考虑：预测s的时候，可能预测对了首，但是没预测对尾，比如图片的例子，“战狼2”的“战”标注为s的首了，
                    但是有可能“2”没有标注为s的尾，那么按照解码规则，会去寻找下一个尾，可能找到了“吴京”的“京”，所以抽出来的实体变成了
                    “战狼2》的主演吴京”。而拿这个s去找o，应该是什么都找不到才对，也就是让模型学习到了负样本，从而排除了这种错误三元组
                    的出现。如果不是这样子的话，根据本文的设计逻辑，找到s后预测o时几乎都能找到一个o（因为训练o时没有负样本），所以遇到
                    这样的情况就找出了一个错误的三元组，降低了pr。因此，这样做我们虽然没有改变recall，但至少提高了precision。
                    '''
                    o1, o2 = np.zeros((len(text), num_classes)), np.zeros((len(text), num_classes))  # 这里变为二维数组是为了把关系和o标签都一起预测
                    for j in items.get((k1, k2), []): # 这里再次印证了上面的注释，当取到不同实体的起始，那么这里对应的结束实体就会为空
                        o1[j[0]][j[2]] = 1
                        o2[j[1]-1][j[2]] = 1
                    S1.append(s1)
                    S2.append(s2)
                    K1.append([k1])
                    K2.append([k2-1])
                    O1.append(o1)
                    O2.append(o2)
                    if len(T1) == self.batch_size or i == idxs[-1]:
                        T1 = seq_padding(T1)
                        T2 = sent2vec(T2)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        O1 = seq_padding(O1, np.zeros(num_classes))
                        O2 = seq_padding(O2, np.zeros(num_classes))
                        K1, K2 = np.array(K1), np.array(K2)
                        yield [T1, T2, S1, S2, K1, K2, O1, O2], None
                        T1, T2, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], [], []

from bert4keras.backend import keras,K
from bert4keras.layers import *
from bert4keras.models import Model
from bert4keras.optimizers import Adam
from keras.callbacks import Callback
import tensorflow as tf
from bert4keras.layers import Lambda, Dense


def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0]) # shape = (K.shape(seq)[0],)  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    batch_idxs = K.expand_dims(batch_idxs, 1) # shape = (K.shape(seq)[0], 1) [ [0], [1], [2], [3], [4], [5], [6], [7], [8],[9] ]
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return tf.gather_nd(seq, idxs) # 按idxs索引seq


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1, keepdims=True)


def dilated_gated_conv1d(seq, mask, dilation_rate=1):
    """膨胀门卷积（残差式）
    """
    dim = K.int_shape(seq)[-1]
    h = Conv1D(dim*2, 3, padding='same', dilation_rate=dilation_rate)(seq)
    def _gate(x):
        dropout_rate = 0.1
        s, h = x
        g, h = h[:, :, :dim], h[:, :, dim:]
        g = K.in_train_phase(K.dropout(g, dropout_rate), g)
        g = K.sigmoid(g)
        return g * s + (1 - g) * h
    seq = Lambda(_gate)([seq, h])
    seq = Lambda(lambda x: x[0] * x[1])([seq, mask])
    return seq


class Attention(Layer):
    """多头注意力机制
    """
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        q_in_dim = input_shape[0][-1]
        k_in_dim = input_shape[1][-1]
        v_in_dim = input_shape[2][-1]
        self.q_kernel = self.add_weight(name='q_kernel',
                                        shape=(q_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.k_kernel = self.add_weight(name='k_kernel',
                                        shape=(k_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.v_kernel = self.add_weight(name='w_kernel',
                                        shape=(v_in_dim, self.out_dim),
                                        initializer='glorot_normal')
    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10
    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = K.dot(q, self.q_kernel) # (None, None, 128) * (128, 128) -> (None, None, 128)
        kw = K.dot(k, self.k_kernel) # (None, None, 128) * (128, 128) -> (None, None, 128)
        vw = K.dot(v, self.v_kernel) # (None, None, 128) * (128, 128) -> (None, None, 128)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.nb_head, self.size_per_head)) # (None, None, 8, 16)
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.nb_head, self.size_per_head)) # (None, None, 8, 16)
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.nb_head, self.size_per_head)) # (None, None, 8, 16)
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3)) # (None, 8, None, 16)
        kw = K.permute_dimensions(kw, (0, 2, 1, 3)) # (None, 8, None, 16)
        vw = K.permute_dimensions(vw, (0, 2, 1, 3)) # (None, 8, None, 16)
        # Attention
        # add me
        Q_seq_reshape = K.reshape(qw,(-1, K.shape(qw)[2], K.shape(qw)[3]))  # (None, 8, None, 16) -> (None, None, 16)
        K_seq_reshape = K.reshape(kw, (-1, K.shape(kw)[2], K.shape(kw)[3]))
        a = K.batch_dot(Q_seq_reshape, K_seq_reshape, axes=[2, 2]) / self.size_per_head ** 0.5  # (None, None, None)
        a = K.reshape(a, (-1, K.shape(qw)[1], K.shape(a)[1], K.shape(a)[2]))  # (1, None, None, None)

        # a = K.batch_dot(qw, kw, [3, 3]) / self.size_per_head**0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = K.softmax(a)
        # 完成输出
        a_reshape = K.reshape(a,(-1, K.shape(a)[2], K.shape(a)[3]))
        vw = K.reshape(vw, (-1, K.shape(vw)[2], K.shape(vw)[3]))
        o = K.batch_dot(a_reshape, vw, [2, 1])
        o = K.reshape(o, (-1, K.shape(a)[1], K.shape(o)[1], K.shape(o)[2]))

        o = K.permute_dimensions( o, (0, 2, 1, 3) )
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


t1_in = Input(shape=(None,)) # 字
t2_in = Input(shape=(None, word_size)) # 词
s1_in = Input(shape=(None,)) # 主实体起始标签
s2_in = Input(shape=(None,)) # 主实体结束标签
k1_in = Input(shape=(1,)) # 送入模型的S的起始位置
k2_in = Input(shape=(1,)) # 送入模型的S的结束位置
o1_in = Input(shape=(None, num_classes)) # 客实体起始标签
o2_in = Input(shape=(None, num_classes)) # 客实体结束标签

t1, t2, s1, s2, k1, k2, o1, o2 = t1_in, t2_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in
mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(t1)

def position_id(x):
    if isinstance(x, list) and len(x) == 2:
        x, r = x
    else:
        r = 0
    pid = K.arange(K.shape(x)[1])
    pid = K.expand_dims(pid, 0)
    pid = K.tile(pid, [K.shape(x)[0], 1])
    return K.abs(pid - K.cast(r, 'int32'))

pid = Lambda(position_id)(t1)
position_embedding = Embedding(maxlen, char_size, embeddings_initializer='zeros')
pv = position_embedding(pid)

t1 = Embedding(len(char2id)+2, char_size)(t1) # 0: padding, 1: unk (None, None, 128)
t2 = Dense(char_size, use_bias=False)(t2) # 词向量也转为同样维度 (None, None, 128)
t = Add()([t1, t2, pv]) # 字向量、词向量、位置向量相加 (None, None, 128)
t = Dropout(0.25)(t)
t = Lambda(lambda x: x[0] * x[1])([t, mask])
t = dilated_gated_conv1d(t, mask, 1) # 12层DGCNN 空洞卷积
t = dilated_gated_conv1d(t, mask, 2)
t = dilated_gated_conv1d(t, mask, 5)
t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 2)
t = dilated_gated_conv1d(t, mask, 5)
t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 2)
t = dilated_gated_conv1d(t, mask, 5)
t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 1)
t_dim = K.int_shape(t)[-1]

# 共享编码层，对应模型图
pn1 = Dense(char_size, activation='relu')(t) # (None, None, 128)
pn1 = Dense(1, activation='sigmoid')(pn1) # (None, None, 1)
pn2 = Dense(char_size, activation='relu')(t)
pn2 = Dense(1, activation='sigmoid')(pn2)
# 抽取S， 对应模型图
h = Attention(8, 16)([t, t, t, mask])
h = Concatenate()([t, h])
h = Conv1D(char_size, 3, activation='relu', padding='same')(h)
ps1 = Dense(1, activation='sigmoid')(h)
ps2 = Dense(1, activation='sigmoid')(h)
ps1 = Lambda(lambda x: x[0] * x[1])([ps1, pn1])
ps2 = Lambda(lambda x: x[0] * x[1])([ps2, pn2])

subject_model = Model([t1_in, t2_in], [ps1, ps2]) # 预测subject的模型

# 全局关系检测模块
t_max = Lambda(seq_maxpool)([t, mask])
pc = Dense(char_size, activation='relu')(t_max)
pc = Dense(num_classes, activation='sigmoid')(pc) # (None, 1, 49)

def get_k_inter(x, n=6):
    seq, k1, k2 = x
    k_inter = [K.round(k1 * a + k2 * (1 - a)) for a in np.arange(n) / (n - 1.)]
    k_inter = [seq_gather([seq, k]) for k in k_inter]
    k_inter = [K.expand_dims(k, 1) for k in k_inter]
    k_inter = K.concatenate(k_inter, 1)
    return k_inter
# S 实体，模型补齐到 6
k = keras.layers.Lambda(get_k_inter, output_shape=(6, t_dim))([t, k1, k2])
k = keras.layers.Bidirectional( keras.layers.GRU(t_dim) )(k) # k = Bidirectional(CuDNNGRU(t_dim))(k) # (None, 256)
k1v = position_embedding(keras.layers.Lambda(position_id)([t, k1])) # (None, None, 128)
k2v = position_embedding(keras.layers.Lambda(position_id)([t, k2])) # (None, None, 128)
kv = Concatenate()([k1v, k2v]) # (None, None, 256)
k = keras.layers.Lambda(lambda x: K.expand_dims(x[0], 1) + x[1])([k, kv]) # (None, None, 256)

h = Attention(8, 16)([t, t, t, mask]) # (None, None, 128)
h = Concatenate()([t, h, k]) # (None, None, 512) 521 = 128+128+256
h = Conv1D(char_size, 3, activation='relu', padding='same')(h) # (None, None, 128)
po = Dense(1, activation='sigmoid')(h) # (None, None, 1)
po1 = Dense(num_classes, activation='sigmoid')(h) # (None, None, 49)
po2 = Dense(num_classes, activation='sigmoid')(h) # (None, None, 49)
po1 = keras.layers.Lambda(lambda x: x[0] * x[1] * x[2] * x[3])([po, po1, pc, pn1]) # (None, None, 49)
po2 = keras.layers.Lambda(lambda x: x[0] * x[1] * x[2] * x[3])([po, po2, pc, pn2])

object_model = Model([t1_in, t2_in, k1_in, k2_in], [po1, po2]) # 输入text和subject，预测object及其关系


train_model = Model([t1_in, t2_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in],
                    [ps1, ps2, po1, po2])

s1 = K.expand_dims(s1, 2)
s2 = K.expand_dims(s2, 2)

s1_loss = K.binary_crossentropy(s1, ps1) # s 起始的label 和 pred
s1_loss = K.sum(s1_loss * mask) / K.sum(mask)
s2_loss = K.binary_crossentropy(s2, ps2) # s 结束的label 和 pred
s2_loss = K.sum(s2_loss * mask) / K.sum(mask)

o1_loss = K.sum(K.binary_crossentropy(o1, po1), 2, keepdims=True)
o1_loss = K.sum(o1_loss * mask) / K.sum(mask)
o2_loss = K.sum(K.binary_crossentropy(o2, po2), 2, keepdims=True)
o2_loss = K.sum(o2_loss * mask) / K.sum(mask)

loss = (s1_loss + s2_loss) + (o1_loss + o2_loss)

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(1e-3))
train_model.summary()


class ExponentialMovingAverage:
    """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    """
    def __init__(self, model, momentum=0.9999):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]
    def inject(self):
        """添加更新算子到model.metrics_updates。
        """
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = K.moving_average_update(w1, w2, self.momentum)
            # self.model.metrics_updates.append(op)
            self.model.metrics_names.append(op)
    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        K.batch_set_value(zip(self.ema_weights, self.old_weights))
    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        ema_weights = K.batch_get_value(self.ema_weights)
        K.batch_set_value(zip(self.model.weights, ema_weights))
    def reset_old_weights(self):
        """恢复模型到旧权重。
        """
        K.batch_set_value(zip(self.model.weights, self.old_weights))


EMAer = ExponentialMovingAverage(train_model)
EMAer.inject()


def extract_items(text_in):
    text_words = tokenize(text_in.lower())
    text_in = ''.join(text_words)
    R = []
    _t1 = [char2id.get(c, 1) for c in text_in]
    _t1 = np.array([_t1])
    _t2 = sent2vec([text_words])
    _k1, _k2 = subject_model.predict([_t1, _t2]) # (1, 102, 1)
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0] # (102,)
    _k1, _k2 = np.where(_k1 > 0.5)[0], np.where(_k2 > 0.4)[0]
    _subjects = []
    for i in _k1:
        j = _k2[_k2 >= i]
        if len(j) > 0:
            j = j[0]
            _subject = text_in[i: j+1]
            _subjects.append((_subject, i, j))
    if _subjects:
        _t1 = np.repeat(_t1, len(_subjects), 0)
        _t2 = np.repeat(_t2, len(_subjects), 0)
        _k1, _k2 = np.array([_s[1:] for _s in _subjects]).T.reshape((2, -1, 1))
        _o1, _o2 = object_model.predict([_t1, _t2, _k1, _k2])
        for i,_subject in enumerate(_subjects):
            _oo1, _oo2 = np.where(_o1[i] > 0.5), np.where(_o2[i] > 0.4)
            for _ooo1, _c1 in zip(*_oo1):
                for _ooo2, _c2 in zip(*_oo2):
                    if _ooo1 <= _ooo2 and _c1 == _c2:
                        _object = text_in[_ooo1: _ooo2+1]
                        _predicate = id2predicate[_c1]
                        R.append((_subject[0], _predicate, _object))
                        break
        zhuanji, gequ = [], []
        for s, p, o in R[:]:
            if p == u'妻子':
                R.append((o, u'丈夫', s))
            elif p == u'丈夫':
                R.append((o, u'妻子', s))
            if p == u'所属专辑':
                zhuanji.append(o)
                gequ.append(s)
        spo_list = set()
        for s, p, o in R:
            if p in [u'歌手', u'作词', u'作曲']:
                if s in zhuanji and s not in gequ:
                    continue
            spo_list.add((s, p, o))
        return list(spo_list)
    else:
        return []


class Evaluate(Callback):
    def __init__(self):
        self.F1 = []
        self.best = 0.
        self.passed = 0
        self.stage = 0
    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，不warmup有不收敛的可能。
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * 1e-3
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
    def on_epoch_end(self, epoch, logs=None):
        EMAer.apply_ema_weights()
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            train_model.save_weights('best_model.weights')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))
        EMAer.reset_old_weights()
        if epoch + 1 == 50 or (
            self.stage == 0 and epoch > 10 and
            (f1 < 0.5 or np.argmax(self.F1) < len(self.F1) - 8)
        ):
            self.stage = 1
            train_model.load_weights('best_model.weights')
            EMAer.initialize()
            K.set_value(self.model.optimizer.lr, 1e-4)
            K.set_value(self.model.optimizer.iterations, 0)
            opt_weights = K.batch_get_value(self.model.optimizer.weights)
            opt_weights = [w * 0. for w in opt_weights]
            K.batch_set_value(zip(self.model.optimizer.weights, opt_weights))
    def evaluate(self):
        orders = ['subject', 'predicate', 'object']
        A, B, C = 1e-10, 1e-10, 1e-10
        F = open('dev_pred.json', 'w')
        for d in tqdm(iter(dev_data)):
            R = set(extract_items(d['text']))
            T = set(d['spo_list'])
            A += len(R & T)
            B += len(R)
            C += len(T)
            s = json.dumps({
                'text': d['text'],
                'spo_list': [
                    dict(zip(orders, spo)) for spo in T
                ],
                'spo_list_pred': [
                    dict(zip(orders, spo)) for spo in R
                ],
                'new': [
                    dict(zip(orders, spo)) for spo in R - T
                ],
                'lack': [
                    dict(zip(orders, spo)) for spo in T - R
                ]
            }, ensure_ascii=False, indent=4)
            F.write(s + '\n')
        F.close()
        return 2 * A / (B + C), A / B, A / C


def test(test_data):
    """输出测试结果
    """
    orders = ['subject', 'predicate', 'object', 'object_type', 'subject_type']
    F = open('test_pred.json', 'w')
    for d in tqdm(iter(test_data)):
        R = set(extract_items(d['text']))
        s = json.dumps({
            'text': d['text'],
            'spo_list': [
                dict(zip(orders, spo + ('', ''))) for spo in R
            ]
        }, ensure_ascii=False)
        F.write(s + '\n')
    F.close()


train_D = data_generator(train_data)
evaluator = Evaluate()


if __name__ == '__main__':
    train_model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=len(train_D),
                              epochs=120,
                              callbacks=[evaluator]
                              )
else:
    train_model.load_weights('best_model.weights')