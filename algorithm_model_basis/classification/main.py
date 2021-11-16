# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/4/8 11:43 
# 文件     ：main.py
# IDE     : PyCharm

from utils.config import *
from utils.data_help import read_csv,cut_sentence,Data_segmentation,read_vocab,label_one_hot,clsc_seq_length,train_corpus,read_label,clean_data
from mode import *
from mode import TextRCNN
from utils.evaluation_metric import Evaluation
from utils.log import *
import numpy as np
import os
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 每次写 log 先清空文件 按需打开或者关闭
# if os.path.exists(config.log_path):
#     f = open(config.log_path, "r+")
#     f.truncate()
def start_log():
    '''
    :return: 程序初始化参数log
    '''
    log('+++'*20)
    log_ = 'model type:{model_type},numclass:{numclass},idtype:{idtype}'.format(model_type=config.model_type, numclass=config.numclass,idtype=config.id_type)
    log(log_)
    log(config.id_type)
class get_train(object):
    def __init__(self):
        pass
    def eval(self,yvalid,predictions):
        eval = Evaluation(yvalid, predictions)
        logloss = eval.multiclass_logloss()
        log("logloss: %0.3f " % logloss)
        acc = eval.acc()
        log('acc: %0.4f' % acc)
        kappa = eval.kappa()
        log('kappa: %0.4f' % kappa)
        class_report = eval.classificaiton_report()
        print(class_report)
        log('Classification report:')
        log('\n' + class_report)
    def LR(self,tfv,xtrain, xvalid, ytrain, yvalid,ch_label_type=False):
        '''
        :param tfv: 类初始化
        :param xtrain:
        :param xvalid:
        :param ytrain:
        :param yvalid:
        :param ch_lael_type:True label 是汉字，需要转换为对应的 label
                            False label 为对应的 label
        :return:
        '''
        if ch_label_type:
            ytrain = tf.argmax(label_one_hot(ytrain,False),axis=1).numpy()
            yvalid = tf.argmax(label_one_hot(yvalid,False),axis=1).numpy()
        if config.id_type == 'tf-idf':
            xtrain_tfv, xvalid_tfv = tfv.tf_idf(xtrain, xvalid)
        elif config.id_type == 'word-count':
            xtrain_tfv, xvalid_tfv = tfv.word_count(xtrain, xvalid)
        elif config.id_type == 'word2vec':
            xtrain_tfv, xvalid_tfv = tfv.word2vec(xtrain, xvalid)
        clf = tfv.LR_train(xtrain_tfv, ytrain)
        predictions = tfv.LR_test(clf, xvalid_tfv)
        self.eval(yvalid,predictions)
    def XGBoost(self,tfv,xtrain, xvalid, ytrain, yvalid,ch_lael_type=False):
        '''
        :param ch_lael_type:True label 是汉字，需要转换为对应的 label
                            False label 为对应的 label
        '''
        if config.ch_label_type:
            ytrain = tf.argmax(label_one_hot(ytrain,False),axis=1).numpy()
            yvalid = tf.argmax(label_one_hot(yvalid,False),axis=1).numpy()
        if config.id_type == 'tf-idf':
            xtrain_tfv, xvalid_tfv = tfv.tf_idf(xtrain, xvalid)
        elif config.id_type == 'word-count':
            xtrain_tfv, xvalid_tfv = tfv.word_count(xtrain, xvalid)
        elif config.id_type == 'word2vec':
            xtrain_tfv, xvalid_tfv = tfv.word2vec(xtrain, xvalid)
        clf = tfv.xgboost_train(xtrain_tfv, ytrain)
        predictions = tfv.xgboost_test(clf, xvalid_tfv)
        self.eval(yvalid,predictions)
    def Textcnn(self,tfv,xtrain, xvalid, ytrain, yvalid, word2id,seq_length):
        '''
        :return:
        '''
        if config.label_multi :
            labels = label_one_hot(ytrain, True)
            yvalid = label_one_hot(yvalid, True)
        else :
            labels = label_one_hot(ytrain, False).numpy()
            yvalid = label_one_hot(yvalid, False).numpy()
        timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
        tfv.train(xtrain, labels, len(word2id), seq_length, config.model_path, timestamp)
        model = load_model(config.model_path)
        predictions = tfv.test(model, xvalid, yvalid)
        self.eval(yvalid,predictions)
    def TextcnnNew(self,tfv,xtrain, xvalid, ytrain, yvalid, word2id,seq_length):
        '''
        :return:
        '''
        if config.label_multi :
            labels = label_one_hot(kwargs['ytrain_word'], True)
            yvalid = label_one_hot(kwargs['yvalid_word'], True)
        else :
            labels = label_one_hot(kwargs['ytrain_word'], False).numpy()
            yvalid = label_one_hot(kwargs['yvalid_word'], False).numpy()
        timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
        tfv.train(xtrain, labels, len(word2id), seq_length, config.model_path, timestamp)
        model = load_model(config.model_path)
        predictions = tfv.test(model, xvalid, yvalid)
        self.eval(yvalid,predictions)
    def TextcnnMultiDim(self,**kwargs):
        tfv = kwargs['tfv']
        if config.label_multi :
            labels = label_one_hot(kwargs['ytrain_word'], True)
            yvalid = label_one_hot(kwargs['yvalid_word'], True)
        else :
            labels = label_one_hot(kwargs['ytrain_word'], False).numpy()
            yvalid = label_one_hot(kwargs['yvalid_word'], False).numpy()
        timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
        log(kwargs['word2id']['<UNK>']+1)
        tfv.train(kwargs['xtrain_char'], kwargs['xtrain_word'], kwargs['xtrain_cx'], labels, len(kwargs['char2id']),
                  kwargs['word2id']['<UNK>']+1, len(kwargs['cx2id']), kwargs['max_length'], config.model_path, timestamp)
        model = load_model(config.model_path)
        predictions = tfv.test(model, kwargs['xvalid_char'], kwargs['xvalid_word'], kwargs['xvalid_cx'], yvalid)
        self.eval(yvalid, predictions)
    def TextRCnn(self,tfv,xtrain_current, xtrain_left, xtrain_right, xvalid_current, xvalid_left, xvalid_right, ytrain, yvalid, word2id,seq_length):
        '''
        :return:
        '''
        if config.label_multi :
            labels = label_one_hot(ytrain, True)
            yvalid = label_one_hot(yvalid, True)
        else :
            labels = label_one_hot(ytrain, False).numpy()
            yvalid = label_one_hot(yvalid, False).numpy()
        timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
        tfv.train(xtrain_current, xtrain_left, xtrain_right, labels, len(word2id), seq_length, config.model_path, timestamp)
        model = load_model(config.model_path)
        predictions = tfv.test(model, xvalid_current, xvalid_left, xvalid_right, yvalid)
        self.eval(yvalid,predictions)
    def TextRNN(self,tfv,xtrain, xvalid, ytrain, yvalid, word2id,seq_lengt):
        '''
        :return:
        '''
        if config.label_multi :
            labels = label_one_hot(ytrain, True)
            yvalid = label_one_hot(yvalid, True)
        else :
            labels = label_one_hot(ytrain, False).numpy()
            yvalid = label_one_hot(yvalid, False).numpy()
        timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
        tfv.train(xtrain, labels, len(word2id), seq_length, config.model_path, timestamp)
        model = load_model(config.model_path)
        predictions = tfv.test(model, xvalid, yvalid)
        self.eval(yvalid,predictions)
    def TextAttBiRNN(self,tfv,xtrain, xvalid, ytrain, yvalid, word2id,seq_lengt):
        '''
        :return:
        '''
        if config.label_multi :
            labels = label_one_hot(ytrain, True)
            yvalid = label_one_hot(yvalid, True)
        else :
            labels = label_one_hot(ytrain, False).numpy()
            yvalid = label_one_hot(yvalid, False).numpy()
        timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
        tfv.train(xtrain, labels, len(word2id), seq_length, config.model_path, timestamp)
        model = load_model(config.model_path)
        predictions = tfv.test(model, xvalid, yvalid)
        self.eval(yvalid,predictions)
if __name__ == '__main__':
    start_log()
    df_origin = read_csv(config.data_path)
    df_origin = df_origin.sample(frac=1)
    get_ob = get_train()
    if config.model_type == 'LR':
        tfv = BasicModels()
        df_origin = cut_sentence(df_origin, 'content')
        xtrain, xvalid, ytrain, yvalid = Data_segmentation(df_origin.content_cut.values, df_origin.label.values)
        if config.ch_label_type:
            get_ob.LR(tfv,xtrain, xvalid, ytrain, yvalid,ch_label_type=True)
        else:
            get_ob.LR(tfv, xtrain, xvalid, ytrain, yvalid)
    elif config.model_type == 'XGBoost':
        tfv = BasicModels()
        df_origin = cut_sentence(df_origin, 'content')
        xtrain, xvalid, ytrain, yvalid = Data_segmentation(df_origin.content_cut.values, df_origin.label.values)
        if config.ch_label_type:
            get_ob.XGBoost(tfv,xtrain, xvalid, ytrain, yvalid,ch_label_type=True)
        else:
            get_ob.LR(tfv, xtrain, xvalid, ytrain, yvalid)
    elif config.model_type == 'TextCnn':
        tfv = TextCnn()
        word2id, id2word = read_vocab(config.vocab_path)
        df_origin = clean_data(df_origin)
        corpus_id ,seq_length = train_corpus(df_origin, word2id)
        label = read_label(df_origin)
        xtrain, xvalid, ytrain, yvalid = Data_segmentation(np.array(corpus_id), np.array(label))
        get_ob.Textcnn(tfv,xtrain, xvalid, ytrain, yvalid, word2id,seq_length)
    elif config.model_type == 'TextCnnNew':
        tfv = TextCnnNew()
        word2id, id2word = read_vocab(config.vocab_path)
        df_origin = clean_data(df_origin)
        corpus_id ,seq_length = train_corpus(df_origin, word2id)
        label = read_label(df_origin)
        xtrain, xvalid, ytrain, yvalid = Data_segmentation(np.array(corpus_id), np.array(label))
        get_ob.TextcnnNew(tfv,xtrain, xvalid, ytrain, yvalid, word2id,seq_length)
    elif config.model_type == 'TextCnnMultiDim': # 多维度编码的Text CNN（字、词、词性）
        tfv = TextCnnMultiDim()
        ## -------------------  数据清洗 + 字典制作 ---------------------
#         df_origin = df_origin.sample(frac=0.3)
        df_origin = clean_data(df_origin)
        char2id, id2char = read_vocab(config.vocab_path) # 字
        df_origin_cut, words, cx = cut_sentence(df_origin, 'content') # 获得切词此表 jieba
        with open(r'data/wv/sentence.txt','w',encoding='utf-8') as f:
            for i in words:
                f.write(i)
                f.write('\n')
        f.close()
        with open(r'data/wv/cx.txt','w',encoding='utf-8') as f:
            for i in cx:
                f.write(i)
                f.write('\n')
        f.close()
        word2id,id2word = read_vocab(r'data/wv/sentence.txt') # 词
        cx2id,id2cx = read_vocab(r'data/wv/cx.txt') # 词性
        log(','.join([str(len(char2id)),str(len(word2id)),str(len(cx2id))]))
        ## ================================================================
        result_char, result_word, result_cx, max_length = train_corpus(df_origin, char2id,word2id,cx2id)
        label = read_label(df_origin)
        xtrain_char, xvalid_char, ytrain_char, yvalid_char = Data_segmentation(np.array(result_char), np.array(label))
        xtrain_word, xvalid_word, ytrain_word, yvalid_word = Data_segmentation(np.array(result_word), np.array(label))
        xtrain_cx, xvalid_cx, ytrain_cx, yvalid_cx = Data_segmentation(np.array(result_cx), np.array(label))
        log(ytrain_char)
        get_ob.TextcnnMultiDim(xtrain_char=xtrain_char, xvalid_char=xvalid_char, ytrain_char=ytrain_char,yvalid_char=yvalid_char,
                  xtrain_word=xtrain_word, xvalid_word=xvalid_word, ytrain_word=ytrain_word, yvalid_word=yvalid_word,
                  xtrain_cx=xtrain_cx, xvalid_cx=xvalid_cx, ytrain_cx=ytrain_cx, yvalid_cx=yvalid_cx,
                  char2id=char2id,word2id=word2id,cx2id=cx2id,max_length=max_length,tfv=tfv)
    elif config.model_type == 'RCNN':
        tfv = TextRCNN()
        word2id, id2word = read_vocab(config.vocab_path)
        df_origin = clean_data(df_origin)
        corpus_id ,seq_length = train_corpus(df_origin, word2id)
        label = read_label(df_origin)
        # 构造 current  left  right
        corpus_id_all = [[0]+j+[0] for j in [i for i in corpus_id]]
        xtrain_current, xvalid_current, ytrain, yvalid = Data_segmentation(np.array(corpus_id), np.array(label))
        xtrain_left, xvalid_left, ytrain, yvalid = Data_segmentation(np.array(corpus_id_all)[...,:-2], np.array(label))
        xtrain_right, xvalid_right, ytrain, yvalid = Data_segmentation(np.array(corpus_id_all)[...,2:], np.array(label))
        get_ob.TextRCnn(tfv,xtrain_current, xtrain_left, xtrain_right, xvalid_current, xvalid_left, xvalid_right, ytrain, yvalid, word2id,seq_length)
    elif config.model_type == 'TextRNN':
        tfv = TextRNN()
        word2id, id2word = read_vocab(config.vocab_path)
        df_origin = clean_data(df_origin)
        corpus_id ,seq_length = train_corpus(df_origin, word2id)
        label = read_label(df_origin)
        xtrain, xvalid, ytrain, yvalid = Data_segmentation(np.array(corpus_id), np.array(label))
        get_ob.TextRNN(tfv,xtrain, xvalid, ytrain, yvalid, word2id,seq_length)
    elif config.model_type == 'TextAttBiRNN':
        tfv = TextAttBiRNN()
        word2id, id2word = read_vocab(config.vocab_path)
        df_origin = clean_data(df_origin)
        corpus_id ,seq_length = train_corpus(df_origin, word2id)
        label = read_label(df_origin)
        xtrain, xvalid, ytrain, yvalid = Data_segmentation(np.array(corpus_id), np.array(label))
        get_ob.TextAttBiRNN(tfv,xtrain, xvalid, ytrain, yvalid, word2id,seq_length)