# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/4/8 11:06 
# 文件     ：config.py
# IDE     : PyCharm
import time
import os


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")


def get_label(path):
    dict_label = {}
    with open(path, 'r') as fp:
        lines = fp.readlines()
    i = 0
    for line in lines:
        dict_label[line.strip()] = i
        i += 1
    return dict_label


# def type_id(str_):
#     if str_ == 'tf-idf':return 'TF_IDF_'
#     if str_ == 'word-count':return 'Word_Count_'
#     if str_ == 'word2vec':return 'Word2Vec_'
#     if str_ == 'TextCnn':return 'TextCNN_'
class config:
    '''
    模型类型
    'LR':'逻辑斯蒂回归'
    'XGBoost':'GBDT'
    'TextCnn':'text cnn model'
    'TextCnnMultiDim':'多维度编码的Text CNN（字、词、词性）'
    'RCNN':'text RCNN model'
    'TextRNN':'text rnn model'
    'TextAttBiRNN':'lstm attention model'
    '''
    model_type = 'TextCnn'

    stop_word = False  # True/False
    id_type = 'word2vec'  # 可选 ['tf-idf','word-count','word2vec']
    numclass = 5
    embedding_size = 300
    batch_size = 256
    epochs = 10
    kernel_size = '3,4,5'
    dropout = 0.5
    regularizers_lambda = 0.001
    num_filters = 100
    ch_label_type = True  # True:汉字label,False:数字label
    label_multi = True  # 是否多标签分类
    loss = 'focal_loss'  # 损失函数  'focal_loss' or ''

    data_path = r'data/corpus.xlsx'  # 训练文本路径
    stop_path = r'data/wv/stopword.txt'  # 停用词表
    word2vec_model_path = r'data/wv/word2vec.model'  # 词向量模型
    embedding_path = r'data/wv/word2vec.model.wv.vectors.npy'  # 词向量文件
    log_path = r'log/log_' + model_type + '.txt'  # 日志文件
    vocab_path = r'data/wv/vocab.txt'  # vocabulary 文件
    model_path = r'model/' + model_type  # 模型存储文件'
    label_path = r'data/wv/tag.txt'  # label 文件
    mkdir(model_path)
    try:
        dict_lable = get_label(label_path)
    except:
        dict_lable = {'其他口号': 0, '工作时间': 1, '要求': 2, '描述': 3, '薪资福利': 4, '公司介绍': 5, '工作地址': 6, '岗位晋升/职业发展': 7, '联系方式': 8,'职位名称':9}
    numclass = len(dict_lable)