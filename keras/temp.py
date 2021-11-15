# _*_coding:utf-8_*_
# 作者     ：
# 创建时间  ：2021/11/11 17:18 
# 文件     ：temp.py
# IDE     : PyCharm
#############################################################################################################

import re
import time
from utility import ProphetClient
import json
import pickle
from collections import defaultdict
import itertools
from itertools import chain
import os
import tensorflow as tf
import pandas as pd
import time
def read_vocab(vocab_file):
    vocab2id = {}
    id2vocab = {}
    for index,line in enumerate([line.strip() for line in open(vocab_file,"r",encoding='utf-8').readlines()]):
        vocab2id[line] = index
        id2vocab[index] = line
    return vocab2id, id2vocab
tag_check = { "I":["B","I"], "E":["B","I"] }
def check_label(front_label,follow_label):
    if not follow_label:
        raise Exception("follow label should not both None")

    if not front_label:
        return True

    if follow_label.startswith("B-"):
        return False

    if (follow_label.startswith("I-") or follow_label.startswith("E-")) and \
        front_label.endswith(follow_label.split("-")[1]) and \
        front_label.split("-")[0] in tag_check[follow_label.split("-")[0]]:
        return True
    return False


def format_result(chars, tags):
    entities = []
    entity = []
    for index, (char, tag) in enumerate(zip(chars, tags)):
        entity_continue = check_label(tags[index - 1] if index > 0 else None, tag)
        if not entity_continue and entity:
            entities.append(entity)
            entity = []
        entity.append([index, char, tag, entity_continue])
    if entity:
        entities.append(entity)
#     print(entities)
    entities_result = []
    for entity in entities:
        if entity[0][2].startswith("B-") or entity[0][2].startswith("S-") :
            entities_result.append(
                {"begin": entity[0][0] + 1,
                 "end": entity[-1][0] + 1,
                 "words": "".join([char for _, char, _, _ in entity]),
                 "type": entity[0][2].split("-")[1]
                 }
            )
    return entities_result
def tokenize(filename,vocab2id,tag2id):
    contents = []
    labels = []
    content = []
    label = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in [elem.strip() for elem in fr.readlines()]:
            try:
                if line != "end":
                    w,t = line.split()
                    if ('\u0041' <= w <='\u005a') or ('\u0061' <= w <='\u007a'):
                        content.append(vocab2id['<ENG>'])
                        label.append(tag2id.get(t,0))
                    elif ('0' <= w <= '9'):
                        content.append(vocab2id['<NUM>'])
                        label.append(tag2id.get(t,0))
                    else:
                        content.append(vocab2id.get(w,vocab2id['<UNK>']))
                        label.append(tag2id.get(t,0))
                else:
                    if content and label:
                        contents.append(content)
                        labels.append(label)
                    content = []
                    label = []
            except Exception as e:
                content = []
                label = []
    contents = tf.keras.preprocessing.sequence.pad_sequences(contents, padding='post')
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, padding='post')
    return contents,labels,len(contents[0])
class Model_utils_data:
    '''
    :数据前处理 后处理 类(父类)
    '''

    def __init__(self):
        pass

    def sentence_id_tfdata(self, sentence_id):
        '''
        return : json 格式的输入文本
        '''
        labels = [0] * len(sentence_id)
        list_input = [{'text': sentence_id, 'labels': labels}]
        dict_input = {'instances': list_input}
        tfdata = json.dumps(dict_input)
        return tfdata

    def model_pred(self, client, name, version, tfdata):
        '''
        :测试模式 client.predict(name, version, tfdata )
        :线上模式client.predict(name, version, tfdata, cluster='prophet-kgmodel' ),需要联系 @川川 确定
        return: resp -> str
        '''
        print(tfdata)
        resp = client.predict(name, version, tfdata, cluster='prophet-web')
        return resp


class Model_JD(Model_utils_data):
    '''
    :模型获取方法类,继承父类Utils_data
    '''

    def __init__(self):
        super(Model_utils_data, self).__init__()
        self.client = ProphetClient()
        self.version = 3
        self.model_name = 'title_parse_v1_20211011'

    def get_parse(self, content):
        '''
        : content : 文本内容
        return: 机构背景
            -1:模型没有预测出结果,-2:模型输入格式错误
        '''
        vocab2id, id2vocab = read_vocab('vocab')
        tag2id, id2tag = read_vocab('tag')
        sentence_id = [vocab2id.get(char, vocab2id['<UNK>']) for char in content]
        print(self.sentence_id_tfdata(sentence_id))
        if len(sentence_id) == 0:
            return -2
        resp = self.model_pred(self.client, self.model_name, self.version, self.sentence_id_tfdata(sentence_id))
        print('预测结果:: ', resp)
        if resp:
            resp = eval(resp)['predictions'][0]['output_3']
            entities_result = format_result(list(content), [id2tag[id] for id in resp])
            return json.dumps(entities_result, indent=4, ensure_ascii=False)
        else:
            return -1


if __name__ == '__main__':
    model = Model_JD()
    while (1):
        content = input("input:")
        mechanism = model.get_parse(content)
        print(mechanism)