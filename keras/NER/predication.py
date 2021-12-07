# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/12/7 14:36 
# 文件     ：predication.py
# IDE     : PyCharm
# import ipdb
import tensorflow as tf
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import open, to_array
import numpy as np
from collections import defaultdict
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dict_path = '../../../../pre_mode/bert-base-cased-ch/vocab.txt'
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
categories = ['上限工作年限', '上限年龄', '下限工作年限', '下限年龄', '专业', '区间工作年限', '区间年龄', '学位', '学历', '学历类型', '学校类型', '技能词', '行业', '语言']
model =  tf.saved_model.load('./1/')
class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, text, threshold=0):
        tokens = tokenizer.tokenize(text, maxlen=256)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        print(token_ids,segment_ids)
        scores = model._default_save_signature([token_ids, segment_ids])  # 根据pb文件自己查看
        print(scores['global_pointer_1'].shape)
        scores = scores['global_pointer_1'].numpy()[0]
#         ipdb.set_trace()
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for l, start, end in zip(*np.where(scores > threshold)):
            entities.append(
                (mapping[start][0], mapping[end][-1], categories[l])
            )
        return entities


NER = NamedEntityRecognizer()
while True:
    text = input("input:")
    entities = NER.recognize(text)
    d = defaultdict(list)
    for e in entities:
        d['entities'].append({
            'start_idx': e[0],
            'end_idx': e[1],
            'type': e[2],
            'entity': text[e[0]:e[1]+1]
        })
    print(d)