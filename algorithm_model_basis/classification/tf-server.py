# _*_coding:utf-8_*_
# 作者     ：
# 创建时间  ：2021/9/28 10:58 
# 文件     ：tf-server.py
# IDE     : PyCharm
# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/02/26 10:50
# 文件     ：tf_serving_http.py
# IDE     : PyCharm

import os
import re
import time
import json

from utility import ProphetClient

import os
import sys
import jieba
import jieba.posseg
import json


class Jd_Content_General(object):
    def __init__(self):
        self.model_params = {'name': 'content_jobtitle_20210726', 'version': 1}
        self.char2id, self.id2char = self.read_vocab(r'data/wv/vocab.txt')
        self.word2id, self.id2word = self.read_vocab(r'data/wv/sentence.txt')  # 词
        self.cx2id, self.id2cx = self.read_vocab(r'data/wv/cx.txt')  # 词性
        self.dict_label2id = self.get_label(r'data/wv/tag_jd描述.txt')
        self.dict_id2label = dict([(v, k) for k, v in self.dict_label2id.items()])
        self.f = open('data/code2ch.json', 'r')
        self.dict_code2ch = json.loads(self.f.read())

    def read_vocab(self, path):
        '''
        :param path: vocab path
        :return: word2id ,id2word
        '''
        with open(path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        word2id, id2word = {}, {}
        word2id = dict([(line.strip(), lines.index(line)) for line in lines])
        id2word = dict([(value, key) for key, value in word2id.items()])
        return word2id, id2word

    def read_vocab(self, path):
        '''
        :param path: vocab path
        :return: word2id ,id2word
        '''
        with open(path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        word2id, id2word = {}, {}
        word2id = dict([(line.strip(), lines.index(line)) for line in lines])
        id2word = dict([(value, key) for key, value in word2id.items()])
        return word2id, id2word

    def train_corpus(self, max_length, line, word2id, sententce2id=None, cx2id=None):
        '''
        :param line:
        :param word2id:字 -》 id
        :param sententce2id:词 -》 id
        :param cx2id:词性 -》 id
        :return: id
        :[2797, 100, 3322, 100, 2523, 100, 1962, 100, 4...]
        '''
        result_char, result_word, result_cx = [], [], []
        # 字
        mid = [word2id[i] if i in word2id.keys() else word2id['[UNK]'] for i in line.replace(' ', '')]
        if len(mid) > max_length:
            mid = mid[0:max_length]
        else:
            mid = mid + [word2id['[PAD]']] * (max_length - len(mid))
        result_char.append(mid)

        # 词
        sen = [sententce2id[i] if i in sententce2id.keys() else sententce2id['<UNK>'] for i in jieba.cut(line)]
        if len(sen) > max_length:
            sen = sen[0:max_length]
        else:
            sen = sen + [sententce2id['<PAD>']] * (max_length - len(sen))
        result_word.append(sen)
        # 词性
        cx = [cx2id[i.flag] if i.flag in cx2id.keys() else cx2id['<UNK>'] for i in jieba.posseg.cut(line)]
        if len(cx) > max_length:
            cx = cx[0:max_length]
        else:
            cx = cx + [cx2id['<PAD>']] * (max_length - len(cx))
        result_cx.append(cx)

        return result_char, result_word, result_cx, max_length

    def get_label(self, path):
        dict_label = {}
        with open(path, 'r') as fp:
            lines = fp.readlines()
        i = 0
        for line in lines:
            dict_label[line.strip()] = i
            i += 1
        return dict_label

    def launcher(self, content):
        """
        content:职位描述和职位要求
        return: 0：审核不通过，1：审核通过
         -1:模型没有预测出结果,-2:模型输入格式错误
        """
        client = ProphetClient()
        content = re.sub(r'[^\u4e00-\u9fa5a-z <>\、\-\_\——\~\—\－\——\,.，。/+()（）0-9]', '', content.lower())
        result_char, result_word, result_cx, max_length = self.train_corpus(438, content, self.char2id, self.word2id,
                                                                            self.cx2id)

        list_input = [{'input_data1': result_char, 'input_data2': result_word, 'input_data3': result_cx}]
        dict_input = {'instances': list_input}
        tfdata = json.dumps(dict_input)
        name = self.model_params['name']
        version = self.model_params['version']
        try:
            resp = client.predict(name, version, tfdata, cluster='prophet-kgmodel')  # prophet-sandbox、prophet-kgmodel

            if resp:
                resp = eval(resp)['predictions']
                result = {}
                for i in range(len(resp[0])):
                    if resp[0][i] > 0.1:
                        result[self.dict_code2ch[self.dict_id2label[i]]] = resp[0][i]
                result = sorted(result.items(), key=lambda result: (result[1], result[0]), reverse=True)
                return result
            else:
                return -1
        except:
            return -2


if __name__ == '__main__':
    test_ob = Jd_Content_General()
    content = "1.梳理中台业务及流程，确认中台和各个系统的边界；2.对接各系统，跟进各业务和中台的业务交互，输出中台能力；3.对中台产品的长期规划，维护稳定；4.有应用数据分析和数据思维能力帮助各业务线拓展新的产品能力；5.负责中台产品规划和设计整合产品中心、CRM、智能进件、账务中心。"
    print(test_ob.launcher(content))
