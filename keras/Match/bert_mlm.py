# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2022/3/30 14:02 
# 文件     ：bert_mlm.py
# IDE     : PyCharm

import json
import numpy as np
from sklearn.metrics import roc_auc_score
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from tqdm import tqdm
import json

min_count = 5
maxlen = 32
batch_size = 128

config_path = 'bert-base-cased-ch/bert_config.json'
checkpoint_path = 'bert-base-cased-ch/bert_model.ckpt'
dict_path = 'bert-base-cased-ch/vocab.txt'

def load_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    f = open(filename, 'r', encoding='utf-8')
    content = f.read()
    f = json.loads(content)['data']
    for l in f:
        a, b, c = l['sen1'], l['sen2'], int(l['label'])
        D.append((a, b, c))
    return D


# 加载数据集
data = load_data(r'gaiic2021.json')
train_data = [d for i,d in enumerate(data) if i % 10 != 0]
valid_data = [d for i,d in enumerate(data) if i % 10 == 0]

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def random_mask(text_ids):
    """随机mask
    """
    input_ids, output_ids = [], []
    rands = np.random.random(len(text_ids))
    for r, i in zip(rands, text_ids):
        if r < 0.15 * 0.8: # mask
            input_ids.append(103)
            output_ids.append(i)
        elif r < 0.15 * 0.9: # 保留
            input_ids.append(i)
            output_ids.append(i)
        elif r < 0.15: # 随机替换,+1为了避开label
            input_ids.append(np.random.choice(len(text_ids)+1))
            output_ids.append(i)
        else:
            input_ids.append(i)
            output_ids.append(0)
    return input_ids, output_ids


def sample_convert(text1, text2, label, random=False):
    """转换为MLM格式
    """
    text1_ids, _ = tokenizer.encode(text1, maxlen=maxlen)
    text2_ids, _  = tokenizer.encode(text2, maxlen=maxlen)
    if random:
        if np.random.random() < 0.5:
            text1_ids, text2_ids = text2_ids, text1_ids
        text1_ids, out1_ids = random_mask(text1_ids)
        text2_ids, out2_ids = random_mask(text2_ids)
    else:
        out1_ids = [0] * len(text1_ids)
        out2_ids = [0] * len(text2_ids)
    token_ids = [101] + text1_ids + [102] + text2_ids + [102]
    segment_ids = [0] * len(token_ids)
#     print([label]+out1_ids+out2_ids)
    output_ids = [label + 1] + out1_ids + [0] + out2_ids + [0]
    return token_ids, segment_ids, output_ids


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids, output_ids = sample_convert(
                text1, text2, label, random
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(output_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [batch_token_ids, batch_segment_ids], batch_output_ids
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


# 加载预训练模型
model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_mlm=True
)


def masked_crossentropy(y_true, y_pred):
    """mask掉非预测部分
    """
    y_true = K.reshape(y_true, K.shape(y_true)[:2])
    y_mask = K.cast(K.greater(y_true, 0.5), K.floatx())
    loss = K.sparse_categorical_crossentropy(y_true, y_pred)
    loss = K.sum(loss * y_mask) / K.sum(y_mask)
    return loss[None, None]


model.compile(loss=masked_crossentropy, optimizer=Adam(1e-5))
model.summary()

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
# test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    """线下评测函数
    """
    Y_true, Y_pred = [], []
    for x_true, y_true in data:
        y_pred = model.predict(x_true)[:, 0, 1:3]
        y_pred = y_pred[:, 1] / (y_pred.sum(axis=1) + 1e-8)
        y_true = y_true[:, 0]
        Y_pred.extend(y_pred)
        Y_true.extend(y_true)
    return roc_auc_score(Y_true, Y_pred)


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_score = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_score = evaluate(valid_generator)
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            model.save_weights('match_data/best_model.weights')
        print(
            u'val_score: %.5f, best_val_score: %.5f\n' %
            (val_score, self.best_val_score)
        )


def predict_to_file(out_file):
    """预测结果到文件
    """
    F = open(out_file, 'w')
    for x_true, _ in tqdm(valid_generator):
        y_pred = model.predict(x_true)[:, 0, 5:7]
        y_pred = y_pred[:, 1] / (y_pred.sum(axis=1) + 1e-8)
        for p in y_pred:
            F.write('%f\n' % p)
    F.close()


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=100,
        callbacks=[evaluator]
    )

else:

    model.load_weights('match_data/best_model.weights')