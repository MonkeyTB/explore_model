# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/12/2 14:40 
# 文件     ：GlobalPointer.py
# IDE     : PyCharm

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.backend import multilabel_categorical_crossentropy
from bert4keras.layers import GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.models import Model
import tensorflow as tf
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

max_len = 256
epochs = 10
batch_size = 4
learning_rate = 2e-5
categories = set()

# bert配置
config_path = '../pre_mode/bert-base-cased-ch/bert_config.json'
checkpoint_path = '../pre_mode/bert-base-cased-ch/bert_model.ckpt'
dict_path = '../pre_mode/bert-base-cased-ch/vocab.txt'

def load_data(filename):
    '''
    加载数据[text,(start, end, label),(start, end, label), ...],

    '''
    res = []
    for d in json.load(open(filename, encoding='utf-8')):
        res.append([d['text']])
        for e in d['entities']:
            start, end, label = e['start_idx'], e['end_idx'], e['type']
            if start <= end:
                res[-1].append((start, end, label))
            categories.add(label)
    return res

# 原始数据
train_data = load_data('data/CMeEE/CMeEE_train.json')  # [['（5）房室结消融和起搏器植入作为反复发作或难治性心房内折返性心动过速的替代疗法。',  (3, 7, 'pro'),  (9, 13, 'pro'), (16, 33, 'dis')], ... ]
valid_data = load_data('data/CMeEE/CMeEE_dev.json')
categories = list(sorted(categories))

# 分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 构造数据
class data_generator(DataGenerator):
    '''
    数据生成器
    '''
    def __iter__(self, random = False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=max_len)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]:i for i,j in enumerate(mapping) if j}
            end_mapping = {j[0]:i for i,j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros((len(categories), max_len, max_len))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    label = categories.index(label)
                    labels[label, start, end] = 1
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, seq_dims=3)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

def global_pointer_crossentropy(y_true, y_pred):
    '''
    给GlobalPoint设计交叉熵
    '''
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))

def global_pointer_f1_score(y_true, y_pred):
    '''
    给GlobalPointer设计F1
    '''
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)


model = build_transformer_model(config_path, checkpoint_path)
# model.input [<tf.Tensor 'Input-Token:0' shape=(None, None) dtype=float32>, <tf.Tensor 'Input-Segment:0' shape=(None, None) dtype=float32>]
# model.output shape=(None, None, 768)
output = GlobalPointer(len(categories), 64)(model.output) # output <tf.Tensor 'global_pointer_1/truediv:0' shape=(None, 9, None, None) dtype=float32>

model = Model(model.input, output)
model.summary()

model.compile(
    loss = global_pointer_crossentropy,
    optimizer = Adam(learning_rate),
    metrics = [global_pointer_f1_score]
)

class NameEntityRecognizer(object):
    '''
    NER 识别器
    '''
    def recognize(self, text, threshold = 0):
        tokens = tokenizer.tokenizer(text, max_len = 512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids  = to_array([token_ids], [segment_ids])
        scores = model.predict([token_ids, segment_ids])[0]
        scores[:, [0,-1]] -= np.inf
        scores[:, :, [0,-1]] -= np.inf
        entities = []
        for l, start, end in zip(*np.where(scores > threshold)):
            entities.append(
                (mapping[start][0], mapping[end][-1], categories[l])
            )
        return entities
NER = NameEntityRecognizer()

def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(NER.recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('cmeee_globalpointer.weights')
            model.save('cmeee_globalpointer.h5')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


def predict_to_file(in_file, out_file):
    """预测到文件
    可以提交到 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
    """
    data = json.load(open(in_file))
    for d in tqdm(data, ncols=100):
        d['entities'] = []
        entities = NER.recognize(d['text'])
        for e in entities:
            d['entities'].append({
                'start_idx': e[0],
                'end_idx': e[1],
                'type': e[2],
                'entity': d['text'][e[0]:e[1]+1]
            })
    json.dump(
        data,
        open(out_file, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)
    '''
    for x,y in train_generator:
        print(x,y)
        break
    len(x) = 2 : token_id, segment_id
    x[0].shape = (4,45)
    len(y) = 4 
    y.shape = (4, 9, 45, 45)
    y[0].shape = (9,45,45)
    4 : batch_size
    9 : label_size
    (45,45) : (text_length, text_length)
    '''

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    model.load_weights('cmeee_globalpointer.weights')
    predict_to_file('data/CMeEE/CMeEE_test.json', 'data/CMeEE/CMeEE_test_pred.json')


''' rematch 函数备注解释说明
is_py2 = six.PY2
def _is_control(ch):
    """控制类字符判断
    """
    return unicodedata.category(ch) in ('Cc', 'Cf') # Cc Cf 类型编码
def _is_special(ch):
    """判断是不是有特殊含义的符号
    """
    return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')   
    
def rematch(self, text, tokens):
    """给出原始的text和tokenize后的tokens的映射关系
    """
    if is_py2: # 判断环境是否是python 2，java不用处理
            text = unicode(text)

    if self._do_lower_case: # tokenizer是否low，java统一转小写就行
            text = text.lower()

    normalized_text, char_mapping = '', [] 
    for i, ch in enumerate(text): # python用法，i表示位置，ch表示对应的字符
            if self._do_lower_case:
                    ch = unicodedata.normalize('NFD', ch) # unicode文本标准化https://blog.csdn.net/weixin_43866211/article/details/98384017
                    ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn']) # 判断unicode是否属于‘Mn’类型https://zhuanlan.zhihu.com/p/93029007
            ch = ''.join([ 
                    c for c in ch
                    if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ]) # 遍历，排除一些特定字符编码值和控制类字符
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

    text, token_mapping, offset = normalized_text, [], 0
    for token in tokens:
            if self._is_special(token): # 特殊含义符号 [CLS] [SEP]
                    token_mapping.append([])
            else:
                    token = self.stem(token)
                    start = text[offset:].index(token) + offset
                    end = start + len(token)
                    token_mapping.append(char_mapping[start:end])
                    offset = end

    return token_mapping
'''