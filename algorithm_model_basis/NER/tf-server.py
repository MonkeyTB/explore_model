# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/8/12 16:06 
# 文件     ：tf-server.py
# IDE     : PyCharm

import re
import time
from utility import ProphetClient
import json
import pickle
from collections import defaultdict
import itertools
from itertools import chain
import os
from utils import tokenize, read_vocab, format_result
from args_help import args
import tensorflow as tf
import pandas as pd
import time
import jieba.posseg


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
        resp = client.predict(name, version, tfdata, cluster='prophet-kgmodel')
        return resp


class Model_JD(Model_utils_data):
    '''
    :模型获取方法类,继承父类Utils_data
    '''

    def __init__(self):
        super(Model_utils_data, self).__init__()
        self.client = ProphetClient()
        self.version = 6
        self.model_name_bg = 'company_parse_20210702'

    def get_mechanism(self, content):
        '''
        : content : 文本内容
        return: 机构背景
            -1:模型没有预测出结果,-2:模型输入格式错误
        '''
        vocab2id, id2vocab = read_vocab(args.vocab_file)
        tag2id, id2tag = read_vocab(args.tag_file)
        sentence_id = [vocab2id.get(char, vocab2id['<UNK>']) for char in content]
        if len(sentence_id) == 0:
            return -2
        resp = self.model_pred(self.client, self.model_name_bg, self.version, self.sentence_id_tfdata(sentence_id))
        #         print(resp)
        if resp:
            resp = eval(resp)['predictions'][0]['output_3']
            print(resp)
            entities_result = format_result(list(content), [id2tag[id] for id in resp])
            return json.dumps(entities_result, indent=4, ensure_ascii=False)
        else:
            return -1


if __name__ == '__main__':
    model = Model_JD()
    content = '四川省广元太星平价大药房连锁有限公司苍溪县唤马四十五药店'
    while (1):
        content = input("input:")
        mechanism = model.get_mechanism(content)
        print(mechanism)
        mechanism = eval(mechanism)

