
import tensorflow as tf
from model import NerModel
from utils import tokenize ,read_vocab ,format_result ,read_wordvocab
import tensorflow_addons as tf_ad
from args_help import args
import json
from lp_pyhanlp import *
from tensorflow.python.platform import gfile
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import re
from zhconv import convert
import unicodedata


vocab2id, id2vocab = read_vocab(args.vocab_file)
tag2id, id2tag = read_vocab(args.tag_file)
embeddings_matrix, word2idx = read_wordvocab(args.fastvec_dir)
def tokenize(text, vocab2id, tag2id, word2id):
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
            content.append(vocab2id.get(w ,vocab2id['<UNK>']))

    if content:
        contents.append(content)

        sententces = re.findall('[a-z0-9]+|[\u4e00-\u9fa5]+|[^a-z0-9\u4e00-\u9fa5]+' ,text)
        word = []
        for s in sententces:
            word.extend([j.word for j in HanLP.segment(s)])


        #             word = [j.word for j in HanLP.segment(text)]
        temp = []
        for i in word:
            temp.extend([i ] *len(i))
        #             print(temp)
        words.append([word2id.get(i, word2id['<UNK>']) for i in temp])


    contents = tf.keras.preprocessing.sequence.pad_sequences(contents, padding='post')
    words = tf.keras.preprocessing.sequence.pad_sequences(words, padding='post')
    return contents, words
def clean(text):
    text = text.lower()  # 大写转小写
    line = convert(text.strip(), 'zh-hans') # 繁体转简体，首位空格去掉
    text = unicodedata.normalize('NFKC', text)
    return text
def one_step():

    ## savemodel
    model =  tf.saved_model.load(args.savemode_dir)
    while True:
        text = input("input:")
        text = clean(text)
        text_sequences, words_text_sequences = tokenize(text, vocab2id, tag2id, word2idx)
        print('char id:', text_sequences)
        print('word id:', words_text_sequences)
        logits, text_lens, log_likelihood, _ = model.call(text_sequences, words_text_sequences, text_sequences)
        paths = []
        for logit, text_len in zip(logits, text_lens):
            viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
            paths.append(viterbi_path)

        entities_result = format_result(list(text), [id2tag[id] for id in paths[0]])
        print(entities_result)
        print(json.dumps(entities_result, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    one_step()