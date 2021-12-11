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
from tqdm import tqdm
import os
import pandas as pd
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
    df = pd.read_excel(filename,sheet_name='标注结果')
    for key,row in df.iterrows():
        content = eval(row.标注结果)
        res.append([row.原始数据])
        if type(content) == dict:
            content = content['remark']
        for e in content:
            start, end, label = e['position'], e['position'] + len(e['text']) - 1, e['type']
            if label == '无效废弃':
                pass
            elif start <= end:
                res[-1].append((start, end, label))
                categories.add(label)
    return res

# 原始数据
train_data = load_data('data/corpus.xlsx')  # [['（5）房室结消融和起搏器植入作为反复发作或难治性心房内折返性心动过速的替代疗法。',  (3, 7, 'pro'),  (9, 13, 'pro'), (16, 33, 'dis')], ... ]
valid_data = train_data[int( 0.8*len(train_data) ):]
train_data = train_data[:int( 0.8*len(train_data) )]
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
output = GlobalPointer(len(categories), 64)(model.output)

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

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    model.load_weights('cmeee_globalpointer.weights')
    predict_to_file('data/CMeEE/CMeEE_test.json', 'data/CMeEE/CMeEE_test_pred.json')