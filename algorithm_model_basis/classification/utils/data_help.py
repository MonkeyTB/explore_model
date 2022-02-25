# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/4/8 11:41 
# 文件     ：data_help.py
# IDE     : PyCharm

import pandas as pd
import jieba
import jieba.posseg
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf
from utils.config import  *
import numpy as np
from utils.log import log
import re
import numpy as np
from lp_pyhanlp import *


def read_csv(path):
    '''
    :param path: csv file path
    :return: dataframe
    '''
    if path.split('.')[1] == 'csv':
        df = pd.read_csv(path,encoding='utf-8')
        df = df.dropna(subset=['content','label'],axis=0)
        return df
    if path.split('.')[1] == 'xlsx':
        df = pd.read_excel(path)
        df.dropna(subset=['content','label'],inplace=True)
        return df

def Q2B(uchar):
    """
       单个字符 全角转半角
       踢除零宽字符
    """
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e: #转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code).replace('\u200b','').replace('\u200c','').replace('\u200d','').replace('\u200e','').replace('\u200f','').replace('\ufeff','').strip()
def stringQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])

def clean_data(df):
    '''
    :param df:
    :return:
    '''
    for key,row in df.iterrows():
        print(key)
        if isinstance(row.content,str):
            row['content'] = re.sub(r'[^\u4e00-\u9fa5a-z <>\、\-\_\——\~\—\－\——\,.，。/+()（）0-9]','',str(row.content).strip().lower())
    df = df[df.content.apply(lambda x: not str(x).isdigit())]
    # df = df[df.content.apply(lambda x: not str(x).isalpha())]
    return df

def cut_sentence(df,type,mode=True):
    '''
    :param df: dataframe
    :mode :True jieba ,默认jieba
          :False HanLP
    :return: dataframe of cut
    '''
    if mode:
        words, cx = [],[]
        df['content_cut'] = df[type].apply(lambda x:jieba.cut(x))
        df['content_cut'] = [' '.join(i) for i in df['content_cut']]
        if config.model_type == 'TextCnnMultiDim':
            words.append('<PAD>')
            cx.append('<PAD>')
            for key,row in df.iterrows():
                c = jieba.posseg.cut(row.content)
                for x in c:
                    words.append(x.word)
                    cx.append(x.flag)
            words = list(set(words))
            cx = list(set(cx))
            words.append('<UNK>')
            cx.append('<UNK>')
            return df, words, cx
        return df
    else:
        words, cx = [],[]
        mode_crf = HanLP.newSegment('crf')
        df['content_cut'] = df[type].apply(lambda x:[i.word for i in mode_crf.seg(x)])
        df['content_cut'] = [' '.join(i) for i in df['content_cut']]
        if config.model_type == 'TextCnnMultiDim':
            words.append('<PAD>')
            cx.append('<PAD>')
            for key,row in df.iterrows():
                c = HanLP.segment(row.content)
                for x in c:
                    words.append(x.word)
                    cx.append(str(x.nature))
            words = list(set(words))
            cx = list(set(cx))
            words.append('<UNK>')
            cx.append('<UNK>')
            return df, words, cx
        return df
def label_transform(df):
    '''
    :param df:
    :return: 文本标签（Text Label）转化为数字(Integer)
    '''
    lbl_enc = preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(df.label.values)
    return y
def Data_segmentation(x,y,test_size = 0.2):
    '''
    :param x:
    :param y:
    :return: 按比例切分训练和测试数据
    '''
    xtrain, xtest, ytrain, ytest = train_test_split(x, y,
#                                                       stratify=y,
                                                      random_state=42,
                                                      test_size=test_size)
    return xtrain, xtest, ytrain, ytest

def read_vocab(path):
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
def clsc_seq_length(xtrain,xvalid):
    '''
    :return: 根据训练和测试样本统计长度，通过不同的方式计算seq_length
    '''
    length = []
    for i in range(xtrain.shape[0]):
        length.append(len(xtrain[i]))
    for i in range(xvalid.shape[0]):
        length.append(len(xvalid[i]))
    return int(np.mean(length) + 2 * np.std(length))


def label_one_hot(lines,label_multi=False,lable_type=True):
    '''
    :param lines:
    :param label_multi:是否多标签分类，True是，Flase不是，默认单标签分类
    :param lable_type:label是否为数字，如果是True，不是False
    :return: one-hot encoder
    '''
    if label_multi:
        dict_label = config.dict_lable
        labels = []
        for line in lines:
            line = line.split(',')
            label = [0]*config.numclass
            for l in line:
                label[dict_label[l.strip()]] = 1
            labels.append(label)
        return np.array(labels)
    else:
        if lable_type:
            labels = []
            for line in lines:
                labels.append(line)
            onehot_label = tf.one_hot(labels,config.numclass)
            return onehot_label
        else:
            dict_label = config.dict_lable
            labels = []
            for line in lines:
                labels.append(dict_label[line])
            onehot_label = tf.one_hot(labels, config.numclass)
            return onehot_label

def train_corpus(df,word2id,sententce2id=None,cx2id=None,mix=True,mode=True):
    '''
    :param df: dataframe
    :param word2id:字 -》 id
    :param sententce2id:词 -》 id
    :param cx2id:词性 -》 id
    :param mix:mix = True,max_length=int(np.mean(length) + 2*np.std(length)),mix=False,max_length=np.max(length)
    :mode :True jieba ,默认jieba
          :False HanLP
    :return: word 转 id
    :[2797, 100, 3322, 100, 2523, 100, 1962, 100, 4...]
    '''
    if not mode:
        mode_crf = HanLP.newSegment('crf')
    length = []
    for key,row in df.iterrows():
        length.append(len(str(row['content'])))
    if mix:
        max_length = int(np.mean(length) + 2*np.std(length))
    else:
        max_length = np.max(length)
    log('max_length:{num}'.format(num=max_length))
    result_char,result_word,result_cx = [],[],[]
    for key,row in df.iterrows():
        # 字
        mid = [word2id[i] if i in word2id.keys() else word2id['[UNK]'] for i in str(row['content']).replace(' ','') ]
        if len(mid) > max_length:
            mid = mid[0:max_length]
        else:
            mid = mid + [word2id['[PAD]']]*(max_length-len(mid))
        result_char.append(mid)
        if config.model_type == 'TextCnnMultiDim':
            # 词
            if mode: # jieba
                sen = [sententce2id[i] if i in sententce2id.keys() else sententce2id['<UNK>'] for i in jieba.cut(row['content'])]
            else:
                sen = [sententce2id[i.word] if i.word in sententce2id.keys() else sententce2id['<UNK>'] for i in mode_crf.seg(row['content'])]
            if len(sen) > max_length:
                sen = sen[0:max_length]
            else:
                sen = sen + [sententce2id['<PAD>']]*(max_length-len(sen))
            result_word.append(sen)
            # 词性
            if mode: # jieba
                cx = [cx2id[i.flag] if i.flag in cx2id.keys() else cx2id['<UNK>'] for i in jieba.posseg.cut(row['content'])]
            else:
                cx = [cx2id[str(i.nature)] if str(i.nature) in cx2id.keys() else cx2id['<UNK>'] for i in mode_crf.seg(row['content'])]
            if len(cx) > max_length:
                cx = cx[0:max_length]
            else:
                cx = cx + [cx2id['<PAD>']] * (max_length - len(cx))
            result_cx.append(cx)
    if config.model_type == 'TextCnnMultiDim':
        return result_char, result_word, result_cx, max_length
    return result_char, max_length
def read_label(df):
    '''
    :param df: dataframe
    :return: label
    '''
    label = []
    for key,row in df.iterrows():
        label.append(row.label.strip())
    return label