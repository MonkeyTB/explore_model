# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2022/3/14 10:04
# 文件     ：prediction.py
# IDE     : PyCharm
import ipdb
import json
import numpy as np
from bert4keras.tokenizers import Tokenizer
import tensorflow as tf
from tqdm import tqdm
from bert4keras.snippets import open, to_array
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dict_path = '../../../../pre_mode/bert-base-cased-ch/vocab.txt'
model = tf.saved_model.load('./1/')
tokenizer = Tokenizer(dict_path, do_lower_case=True)
predicate2id = {'1': 1, '2': 2, '3': 3, '4': 4}
id2predicate = dict([(v, k) for k, v in predicate2id.items()])


def extract_spoes(text):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, maxlen=128)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=128)  # 1*48   1*48
    subject_labels = np.zeros((len(token_ids), 2), dtype=int)  # 1*48*2
    subject_ids = np.zeros(2, dtype=int)  # 48*2
    object_labels = np.zeros((len(token_ids), 4, 2), dtype=int)  # 1*48*4*2
    token_ids, segment_ids, subject_labels, subject_ids, object_labels = to_array([token_ids], [segment_ids],
                                                                                  [subject_labels], [subject_ids],
                                                                                  [object_labels])
    # 抽取subject

    subject_preds = model._default_save_signature([token_ids, segment_ids, subject_labels, subject_ids, object_labels])
    subject_preds = subject_preds['total_loss_1'].numpy()
    subject_preds[:, [0, -1]] *= 0
    start = np.where(subject_preds[0, :, 0] > 0.6)[0]
    end = np.where(subject_preds[0, :, 1] > 0.6)[0]

    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
    if subjects:
        spoes = []
        # 传入subject，抽取object和predicate
        #         ipdb.set_trace()
        subject_labels = np.zeros((len(token_ids[0]), 2))
        for s in range(subject_ids.shape[0]):
            subject_labels[subjects[s], 0] = 1
            subject_labels[subjects[s], 1] = 1
        subject_ids = np.array(subjects)
        object_labels = np.zeros((len(token_ids[0]), 4, 2), dtype=int)

        #         ipdb.set_trace()
        token_ids = np.repeat(token_ids, len(subjects), 0)  # 2 * 48
        segment_ids = np.repeat(segment_ids, len(subjects), 0)  # 2 * 48
        subject_labels = np.repeat([subject_labels], len(subjects), 0)  # 2 * 48 * 2
        object_labels = np.repeat([object_labels], len(subjects),
                                  0)  # 2(subjects) * 48(token_ids) * 4(label num) * 2(start end)

        #         ipdb.set_trace()
        object_preds = model._default_save_signature(
            [token_ids, segment_ids, subject_labels, subject_ids, object_labels])
        object_preds = object_preds['total_loss_1_1'].numpy()
        object_preds[:, [0, -1]] *= 0
        for subject, object_pred in zip(subject_ids, object_preds):
            start = np.where(object_pred[:, :, 0] > 0.5)
            end = np.where(object_pred[:, :, 1] > 0.5)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append(
                            ((mapping[subject[0]][0],
                              mapping[subject[1]][-1]), predicate1,
                             (mapping[_start][0], mapping[_end][-1]))
                        )
                        break
        return [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1])
                for s, p, o, in spoes]
    else:
        return []


def tag(res):
    '''对结果按关系进行拼接
    '''
    sort_res = sorted(res, key=lambda x: (x[1], x[0], x[2]), reverse=True)
    c = []
    del_ = []  # 记录和‘2’搭配的‘1’关系的 index
    for res in sort_res:
        if res[1] == '2':
            for i, mid in enumerate(sort_res):
                if mid[2] == res[0]:
                    c.append(mid[0] + res[0] + res[2])
                    del_.append(i)

    sort_res = [x for i, x in enumerate(sort_res) if i not in del_]

    for res in sort_res:
        c.append(res[0] + res[2])
    return c


while 1:
    text = input('input:')
    res = extract_spoes(text)
    print(res)
    print(tag(res))

