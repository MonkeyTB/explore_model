#############################################################################################################

import re
import time
from utility import ProphetClient
import json
import pickle
from collections import defaultdict
import itertools
from itertools import chain
from lp_pyhanlp import *
import os
from utils import tokenize, read_vocab, format_result, read_wordvocab
from args_help import args
import tensorflow as tf
import pandas as pd
import time


class Model_utils_data:
    '''
    :数据前处理 后处理 类(父类)
    '''

    def __init__(self):
        pass

    def sentence_id_tfdata(self, text_sequences, words_text_sequences):
        '''
        return : json 格式的输入文本
        '''
        labels = [0] * len(text_sequences[0])
        print(text_sequences, words_text_sequences, labels)
        list_input = [{'text': text_sequences, 'sentence': words_text_sequences, 'labels': [labels]}]
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
        resp = client.predict(name, version, tfdata, cluster='prophet-graph-sandbox')
        return resp


class Model_JD(Model_utils_data):
    '''
    :模型获取方法类,继承父类Utils_data
    '''

    def __init__(self):
        super(Model_utils_data, self).__init__()
        self.client = ProphetClient()
        self.version = 5
        self.model_name = 'title_parse_v1_20211011'

    def tokenize(self, text, vocab2id, tag2id, word2id):
        contents = []
        labels = []
        words = []

        content = []
        label = []
        for w in text:
            #             if ('\u0041' <= w <='\u005a') or ('\u0061' <= w <='\u007a'):
            #                 content.append(vocab2id['<ENG>'])
            if ('0' <= w <= '9'):
                content.append(vocab2id['<NUM>'])
            else:
                content.append(vocab2id.get(w, vocab2id['<UNK>']))

        if content:
            contents.append(content)

            sententces = re.findall('[a-z0-9]+|[\u4e00-\u9fa5]+|[^a-z0-9\u4e00-\u9fa5]+', text)
            word = []
            for s in sententces:
                word.extend([j.word for j in HanLP.segment(s)])

            #             word = [j.word for j in HanLP.segment(text)]
            temp = []
            for i in word:
                temp.extend([i] * len(i))
            #             print(temp)
            words.append([word2id.get(i, word2id['<UNK>']) for i in temp])

        #         contents = tf.keras.preprocessing.sequence.pad_sequences(contents, padding='post')
        #         words = tf.keras.preprocessing.sequence.pad_sequences(words, padding='post')
        return contents, words

    def get_parse(self, content):
        '''
        : content : 文本内容
        return: 机构背景
            -1:模型没有预测出结果,-2:模型输入格式错误
        '''
        vocab2id, id2vocab = read_vocab(args.vocab_file)
        tag2id, id2tag = read_vocab(args.tag_file)
        embeddings_matrix, word2idx = read_wordvocab(args.fastvec_dir)
        text_sequences, words_text_sequences = self.tokenize(content, vocab2id, tag2id, word2idx)

        resp = self.model_pred(self.client, self.model_name, self.version,
                               self.sentence_id_tfdata(text_sequences, words_text_sequences))
        print(resp)
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











