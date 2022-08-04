import json
import numpy as np
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from keras.losses import kullback_leibler_divergence as kld
from tqdm import tqdm

num_classes = 119
max_len = 128
batch_size = 32

# load pre model
config_path = 'D:/下载/chinese_roberta_L-4_H-312_A-12_K-104/bert_config.json'
checkpoint_path = 'D:/下载/chinese_roberta_L-4_H-312_A-12_K-104/bert_model.ckpt'
dict_path = 'D:/下载/chinese_roberta_L-4_H-312_A-12_K-104/vocab.txt'


def load_data(file_name):
    """
    加载数据
    :param file_name: 文件路径
    :return: （文本，标签id）
    """
    D = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label = l['sentence'], l['label']
            D.append((text, int(label)))
    return D


# load data file
train_data = load_data('data/train.json')
valid_data = load_data('data/dev.json')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class dataGenerator(DataGenerator):
    """
    数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=max_len)
            for i in range(2):
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


train_generator = dataGenerator(train_data, batch_size)
valid_generator = dataGenerator(valid_data, batch_size)

bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    dropout_rate=0.3,
    return_keras_model=False)

out_put = Lambda(lambda x: x[:, 0])(bert.model.output)
out_put = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(out_put)
model = keras.models.Model(bert.model.input, out_put)
model.summary()


# 自定义loss函数
def cross_entropy_with_r_drop(y_true, y_pre, alpha=4):
    """
    配合R-Drop的交叉熵损失
    :param y_true:
    :param y_pre:
    :param alpha:
    :return:
    """
    y_true = K.reshape(y_true, K.shape(y_pre)[:-1])
    y_true = K.cast(y_true, 'int32')
    loss_1 = K.mean(K.sparse_categorical_crossentropy(y_true, y_pre))
    loss_2 = kld(y_pre[::2], y_pre[1::2]) + kld(y_pre[1::2], y_pre[::2])
    return loss_1 + K.mean(loss_2) / 4 * alpha


model.compile(
    loss=cross_entropy_with_r_drop,
    optimizer=Adam(2e-5),
    metrics=['sparse_categorical_accuracy']
)


def evaluate(data):
    """
    评测准确率
    :param data:
    :return:
    """
    total, right = 0, 0
    for x_true, y_true in data:
        y_pre = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_pre == y_true).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """
    评估与保存
    """

    def __init__(self):
        super().__init__()
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        """
        :param epoch:
        :param logs:
        :return:
        """
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save('model/CRDP.h5')
        print(u'val_acc: %.5f, best_val_acc: %.5f\n' % (val_acc, self.best_val_acc))


def pre_to_file(in_file, out_file):
    """
    输出预测文件
    :param in_file:
    :param out_file:
    :return:
    """
    f_out = open(out_file, 'w')
    with open(in_file, 'r') as f:
        for l in tqdm(f):
            l = json.load(l)
            text = l['text']
            token_ids, segment_ids = tokenizer.encode(text)
            label = model.predict([token_ids], [segment_ids])[0].argmax()
            l = json.dumps({'id': str(l['id']), 'label': str(label)})
            f_out.write(l + '\n')
        f_out.close()


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=50,
        callbacks=[evaluator]
    )
