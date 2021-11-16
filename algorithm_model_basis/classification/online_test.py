########################## 字词词性 ############################

# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/4/12 17:16
# 文件     ：online_test.py
# IDE     : PyCharm

from utils.config import *
from tensorflow.keras.models import load_model
from utils.data_help import *
from pandas import DataFrame

def train_corpus(max_length,line,word2id,sententce2id=None,cx2id=None):
    '''
    :param line:
    :param word2id:字 -》 id
    :param sententce2id:词 -》 id
    :param cx2id:词性 -》 id
    :return: id
    :[2797, 100, 3322, 100, 2523, 100, 1962, 100, 4...]
    '''
    result_char,result_word,result_cx = [],[],[]
    # 字
    mid = [word2id[i] if i in word2id.keys() else word2id['[UNK]'] for i in line.replace(' ','') ]
    if len(mid) > max_length:
        mid = mid[0:max_length]
    else:
        mid = mid + [word2id['[PAD]']]*(max_length-len(mid))
    result_char.append(mid)
    if config.model_type == 'TextCnnMultiDim':
        # 词
        sen = [sententce2id[i] if i in sententce2id.keys() else sententce2id['<UNK>'] for i in jieba.cut(line)]
        if len(sen) > max_length:
            sen = sen[0:max_length]
        else:
            sen = sen + [sententce2id['<PAD>']]*(max_length-len(sen))
        result_word.append(sen)
        # 词性
        cx = [cx2id[i.flag] if i.flag in cx2id.keys() else cx2id['<UNK>'] for i in jieba.posseg.cut(line)]
        if len(cx) > max_length:
            cx = cx[0:max_length]
        else:
            cx = cx + [cx2id['<PAD>']] * (max_length - len(cx))
        result_cx.append(cx)
    if config.model_type == 'TextCnnMultiDim':
        return result_char, result_word, result_cx, max_length
    return result_char, max_length


model = load_model(config.model_path)
char2id, id2char = read_vocab(config.vocab_path)
word2id,id2word = read_vocab(r'data/wv/sentence.txt') # 词
cx2id,id2cx = read_vocab(r'data/wv/cx.txt') # 词性
content,label,mark = [],[],[]
dict_id2label = dict([(v,k) for k,v in config.dict_lable.items()])
results,labels = [], []
################################################## text ########################################
with open('data/title.txt','r') as pf:
    lines=pf.readlines()
for line in lines:
    line = re.sub(r'[^\u4e00-\u9fa5a-z <>\、\-\_\——\~\—\－\——\,.，。/+()（）0-9]','',line.strip().lower())
    content.append(line)
    check = re.sub(r'[^a-z]','',line.strip().lower())
    if len(check)/(len(line)+0.000001) > 0.75:
        if config.model_type == 'TextCnnMultiDim':
            result_char, result_word, result_cx, max_length = train_corpus(10,line,char2id,word2id,cx2id)
            y_pred_one_hot = model.predict(x=[np.array(result_char), np.array(result_word), np.array(result_cx)], batch_size=1, verbose=0)
        elif config.model_type == 'TextCnn':
            result_char,max_length = train_corpus(77,line,char2id)
            y_pred_one_hot = model.predict(x=np.array(result_char),batch_size=1,verbose=0)
        results.append(['其他口号'])
        y_pred = tf.math.argmax(y_pred_one_hot, axis=1).numpy()[0]
        if dict_id2label[y_pred] == '职位名称': labels.append('其他口号')
        else: labels.append(dict_id2label[y_pred])
    else:
        if config.model_type == 'TextCnnMultiDim':
            result_char, result_word, result_cx, max_length = train_corpus(10,line,char2id,word2id,cx2id)
            y_pred_one_hot = model.predict(x=[np.array(result_char), np.array(result_word), np.array(result_cx)], batch_size=1, verbose=0)
        elif config.model_type == 'TextCnn':
            result_char,max_length = train_corpus(77,line,char2id)
            y_pred_one_hot = model.predict(x=np.array(result_char),batch_size=1,verbose=0)
        result = {}
        for i in range(len(y_pred_one_hot[0])):
            result[dict_id2label[i]] = y_pred_one_hot[0][i]
        result = sorted(result.items(), key = lambda result:(result[1], result[0]),reverse=True)
        results.append(result)
        y_pred = tf.math.argmax(y_pred_one_hot, axis=1).numpy()[0]
        if dict_id2label[y_pred] == '工作时间': labels.append('其他口号')
        elif dict_id2label[y_pred] == '公司介绍': labels.append('其他口号')
        elif dict_id2label[y_pred] == '岗位晋升/职业发展': labels.append('薪资福利')
        elif dict_id2label[y_pred] == '联系方式': labels.append('其他口号')
        else: labels.append(dict_id2label[y_pred])
c = {'content':content,'result':results,'label':labels}
df = DataFrame(c)
df.to_csv('data/online_title_pred.csv',encoding='utf-8')
#################################### csv ##################################################
results, labels, content = [], [], []
online_path = r'data/online_test.csv'
df_online = pd.read_csv(online_path,encoding='utf-8')
for key,row in df_online.iterrows():
    line = re.sub(r'[^\u4e00-\u9fa5a-z <>\、\-\_\——\~\—\－\——\,.，。/+()（）0-9]','',row.content.strip().lower())
    content.append(line)
    check = re.sub(r'[^a-z]','',row.content.strip().lower())
    if len(check)/(len(line)+0.000001) > 0.75:
        if config.model_type == 'TextCnnMultiDim':
            result_char, result_word, result_cx, max_length = train_corpus(10,line,char2id,word2id,cx2id)
            y_pred_one_hot = model.predict(x=[np.array(result_char), np.array(result_word), np.array(result_cx)], batch_size=1, verbose=0)
        elif config.model_type == 'TextCnn':
            result_char,max_length = train_corpus(77,line,char2id)
            y_pred_one_hot = model.predict(x=np.array(result_char),batch_size=1,verbose=0)
        results.append(['其他口号'])
        y_pred = tf.math.argmax(y_pred_one_hot, axis=1).numpy()[0]
        if dict_id2label[y_pred] == '职位名称': labels.append('其他口号')
        else: labels.append(dict_id2label[y_pred])
    else:
        if config.model_type == 'TextCnnMultiDim':
            result_char, result_word, result_cx, max_length = train_corpus(10,line,char2id,word2id,cx2id)
            y_pred_one_hot = model.predict(x=[np.array(result_char), np.array(result_word), np.array(result_cx)], batch_size=1, verbose=0)
        elif config.model_type == 'TextCnn':
            result_char,max_length = train_corpus(77,line,char2id)
            y_pred_one_hot = model.predict(x=np.array(result_char),batch_size=1,verbose=0)
        result = {}
        for i in range(len(y_pred_one_hot[0])):
            result[dict_id2label[i]] = y_pred_one_hot[0][i]
        result = sorted(result.items(), key = lambda result:(result[1], result[0]),reverse=True)
        results.append(result)
        y_pred = tf.math.argmax(y_pred_one_hot, axis=1).numpy()[0]
        if dict_id2label[y_pred] == '工作时间': labels.append('其他口号')
        elif dict_id2label[y_pred] == '公司介绍': labels.append('其他口号')
        elif dict_id2label[y_pred] == '岗位晋升/职业发展': labels.append('薪资福利')
        elif dict_id2label[y_pred] == '联系方式': labels.append('其他口号')
        else: labels.append(dict_id2label[y_pred])


c = {'content':content,'result':results,'label':labels}
df = DataFrame(c)
df.to_csv('data/online_pred.csv',encoding='utf-8')

def one_test():
    while(1):
        content = input('please input:')
        line = re.sub(r'[^\u4e00-\u9fa5a-z <>\、\-\_\——\~\—\－\——\,.，。/+()（）0-9]','',content.strip().lower())
        check = re.sub(r'[^a-z]','',content.strip().lower())
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',len(check),len(line),len(check)/(len(line)+0.000001),len(content) )
        if len(check)/(len(line)+0.000001) > 0.8:
            if config.model_type == 'TextCnnMultiDim':
                result_char, result_word, result_cx, max_length = train_corpus(10,line,char2id,word2id,cx2id)
                y_pred_one_hot = model.predict(x=[np.array(result_char), np.array(result_word), np.array(result_cx)], batch_size=1, verbose=0)
            elif config.model_type == 'TextCnn':
                result_char,max_length = train_corpus(77,line,char2id)
                y_pred_one_hot = model.predict(x=np.array(result_char),batch_size=1,verbose=0)
            results.append(['其他口号'])
            y_pred = tf.math.argmax(y_pred_one_hot, axis=1).numpy()[0]
            if dict_id2label[y_pred] == '职位名称': labels.append('其他口号')
            else: labels.append(dict_id2label[y_pred])
            print(labels,dict_id2label[y_pred])
        else:
            if config.model_type == 'TextCnnMultiDim':
                result_char, result_word, result_cx, max_length = train_corpus(10,line,char2id,word2id,cx2id)
                y_pred_one_hot = model.predict(x=[np.array(result_char), np.array(result_word), np.array(result_cx)], batch_size=1, verbose=0)
            elif config.model_type == 'TextCnn':
                result_char,max_length = train_corpus(77,line,char2id)
                y_pred_one_hot = model.predict(x=np.array(result_char),batch_size=1,verbose=0)
            result = {}
            for i in range(len(y_pred_one_hot[0])):
                result[dict_id2label[i]] = y_pred_one_hot[0][i]
            result = sorted(result.items(), key = lambda result:(result[1], result[0]),reverse=True)
            results.append(result)
            y_pred = tf.math.argmax(y_pred_one_hot, axis=1).numpy()[0]
            if dict_id2label[y_pred] == '工作时间': labels.append('其他口号')
            elif dict_id2label[y_pred] == '公司介绍': labels.append('其他口号')
            elif dict_id2label[y_pred] == '岗位晋升/职业发展': labels.append('薪资福利')
            elif dict_id2label[y_pred] == '联系方式': labels.append('其他口号')
            else: labels.append(dict_id2label[y_pred])
            print(labels)
if __name__ == '__main__':
    one_test()




