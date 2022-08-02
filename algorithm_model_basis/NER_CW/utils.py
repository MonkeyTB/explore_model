import tensorflow as tf
import json, os
from my_log import logger
from lp_pyhanlp import *
from gensim.models.fasttext import FastText
import numpy as np
from zhconv import convert
import re


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        logger.info("build %s file" % path)

    else:
        logger.info("---  There is %s folder!  ---" % path)


def read_wordvocab(file):
    '''
    :加载预训练词向量
    '''
    model_name = file
    embedding_mode = FastText.load(model_name)
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    word_list = [(k, embedding_mode.wv[k]) for k, v in embedding_mode.wv.key_to_index.items()]
    embeddings_matrix = np.zeros((len(embedding_mode.wv.key_to_index) + 2, embedding_mode.vector_size))
    for i in range(len(word_list)):
        word = word_list[i][0]
        word2idx[word] = i + 2
        embeddings_matrix[i + 2] = word_list[i][1]
    with open(r'data/word_vocab.txt', 'w') as fp:
        for key, value in word2idx.items():
            fp.write(key)
            fp.write('\n')
    return embeddings_matrix, word2idx


def build_vocab(corpus_file_list, vocab_file, tag_file):
    words = set()
    tags = set()
    for file in corpus_file_list:
        for line in open(file, "r", encoding='utf-8').readlines():
            line = convert(line.strip(), 'zh-hans')
            if line == "end":
                continue
            try:
                w, t = line.split()
                w = w.lower()
                #                 if ('\u0041' <= w <='\u005a') or ('\u0061' <= w <='\u007a'):
                #                     words.add('<ENG>')
                #                     tags.add(t)
                if ('0' <= w <= '9'):
                    words.add('<NUM>')
                    tags.add(t)
                else:
                    words.add(w)
                    tags.add(t)
            except Exception as e:
                print(line.split())
                # raise e

    if not os.path.exists(vocab_file):
        with open(vocab_file, "w") as f:
            for index, word in enumerate(["<PAD>", "<UNK>"] + list(words)):
                f.write(word + "\n")

    tag_sort = {
        "O": 0,
        "B": 1,
        "I": 2,
        "E": 3,
        "S": 4
    }

    tags = sorted(list(tags),
                  key=lambda x: (len(x.split("-")), x.split("-")[-1], tag_sort.get(x.split("-")[0], 100))
                  )
    if not os.path.exists(tag_file):
        with open(tag_file, "w") as f:
            for index, tag in enumerate(["<UNK>"] + tags):
                f.write(tag + "\n")


# build_vocab(["./data/train.utf8","./data/test.utf8"])


def read_vocab(vocab_file):
    vocab2id = {}
    id2vocab = {}
    for index, line in enumerate([line.strip() for line in open(vocab_file, "r", encoding='utf-8').readlines()]):
        vocab2id[line] = index
        id2vocab[index] = line
    return vocab2id, id2vocab


# print(read_vocab("./data/tags.txt"))


# 检验是否全是中文字符
def is_all_eng(strs):
    for _char in strs:
        if not 'a' <= _char <= 'z' and not '1' <= _char <= '9':
            return False
    return True


def tokenize(filename, vocab2id, tag2id, word2id, type):
    '''
    type: 1->字词向量    0->字向量
    '''
    contents = []
    labels = []
    words = []

    content = []
    label = []
    with open(filename, 'r', encoding='utf-8') as fr:
        sententce = ''
        for line in [elem.strip() for elem in fr.readlines()]:
            try:
                if line != "end":
                    w, t = line.split()
                    w = w.lower()
                    sententce += line.split()[0]
                    #                     if ('\u0041' <= w <='\u005a') or ('\u0061' <= w <='\u007a'):
                    #                         content.append(vocab2id['<ENG>'])
                    #                         label.append(tag2id.get(t,0))
                    if ('0' <= w <= '9'):
                        content.append(vocab2id['<NUM>'])
                        label.append(tag2id.get(t, 0))
                    else:
                        content.append(vocab2id.get(w, vocab2id['<UNK>']))
                        label.append(tag2id.get(t, 0))
                else:
                    if content and label:
                        contents.append(content)
                        labels.append(label)
                        sententces = re.findall('[a-z0-9]+|[\u4e00-\u9fa5]+|[^a-z0-9\u4e00-\u9fa5]+', sententce)
                        word = []
                        for s in sententces:
                            if not is_all_eng(s):
                                word.extend([j.word for j in HanLP.segment(s)])
                            else:
                                word.extend([s])

                        temp = []
                        for i in word:
                            temp.extend([i] * len(i))
                        words.append([word2id.get(i, word2id['<UNK>']) for i in temp])
                        if len(content) != len(label) or len(content) != len(
                                [word2id.get(i, word2id['<UNK>']) for i in temp]):
                            print(sententce)
                            print(len(content), len(label), len([word2id.get(i, word2id['<UNK>']) for i in temp]))
                    sententce = ''
                    content = []
                    label = []

            except Exception as e:
                content = []
                label = []
                word = []
                sententce = ''

    print(len(contents), len(labels), len(words))
    contents = tf.keras.preprocessing.sequence.pad_sequences(contents, padding='post')
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, padding='post')
    words = tf.keras.preprocessing.sequence.pad_sequences(words, padding='post')
    return contents, words, labels, len(contents[0])


def sentence2id(content, vocab2id):
    '''
    content 转换为id
    '''
    ids = []
    for s in content:
        if ('\u0041' <= w <= '\u005a') or ('\u0061' <= w <= '\u007a'):
            ids.append(vocab2id['<ENG>'])
        elif ('0' <= w <= '9'):
            ids.append(vocab2id['<NUM>'])
        elif w in vocab2id.keys():
            ids.append(vocab2id[w])
        else:
            ids.append(vocab2id.get(w, vocab2id['<UNK>']))
    return ids


tag_check = {
    "I": ["B", "I"],
    "E": ["B", "I"]
}


def check_label(front_label, follow_label):
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
        if entity[0][2].startswith("B-") or entity[0][2].startswith("S-"):
            entities_result.append(
                {"begin": entity[0][0] + 1,
                 "end": entity[-1][0] + 1,
                 "words": "".join([char for _, char, _, _ in entity]),
                 "type": entity[0][2].split("-")[1]
                 }
            )
    return entities_result


if __name__ == "__main__":
    text = ['四', '川', '省', '广', '元', '太', '星', '平', '价', '大', '药', '房', '连', '锁', '有', '限', '公', '司', '苍', '溪', '县',
            '唤', '马', '四', '十', '五', '药', '店']
    tags = ['B-localtion', 'I-localtion', 'E-localtion', 'B-kernel', 'I-kernel', 'I-kernel', 'E-kernel', 'B-industry',
            'I-industry', 'B-organization', 'I-organization', 'E-organization', 'O', 'O', 'O', 'O', 'O',
            'E-organization', 'B-2_localtion', 'I-2_localtion', 'I-2_localtion', 'I-2_localtion', 'I-2_kernel',
            'I-2_kernel', 'E-2_kernel', 'O', 'B-2_industry', 'E-2_industry']
    print(len(text), len(tags))
    entities_result = format_result(text, tags)
    print(json.dumps(entities_result, indent=4, ensure_ascii=False))

