# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2022/3/21 13:39 
# 文件     ：prediction.py
# IDE     : PyCharm
import tensorflow as tf
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import open, to_array
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

dict_path = '../../../../pre_mode/bert-base-cased-ch/vocab.txt'
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
maxlen = 128
model =  tf.saved_model.load('./1/')
dict_id2label = {0:'other', 1:'title', 2:'require', 3:'describe', 4:'salary'}
class Theseus(object):
    """theseus
    """
    def recognize(self, text):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        token_ids, segment_ids = to_array([token_ids]), to_array([segment_ids])
        scores = model._default_save_signature([token_ids, segment_ids])
        tag = dict_id2label[scores['model_3'].numpy().argmax(axis=1)[0]]
        return tag


BTH = Theseus()
while True:
    text = input("input:")
    classificate = BTH.recognize(text)
    print(classificate)