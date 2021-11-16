# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/5/21 14:17
# 文件     ：utils.py
# IDE     : PyCharm
import tensorflow as tf
import json,os
from my_log import logger

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        logger.info("build %s file" % path)

    else:
        logger.info("---  There is %s folder!  ---" % path)
def build_vocab(corpus_file_list, vocab_file, tag_file):
    words = set()
    tags = set()
    for file in corpus_file_list:
        for line in open(file, "r", encoding='utf-8').readlines():
            line = line.strip()
            if line == "end":
                continue
            try:
                w,t = line.split()
                words.add(w)
                tags.add(t)
            except Exception as e:
                print(line.split())
                # raise e

    if not os.path.exists(vocab_file):
        with open(vocab_file,"w") as f:
            for index,word in enumerate(['<PAD>','<UNK>']+list(words) ):
                f.write(word+"\n")

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
        with open(tag_file,"w") as f:
            for index,tag in enumerate(["<UNK>"]+tags):
                f.write(tag+"\n")

# build_vocab(["./data/train.utf8","./data/test.utf8"])


def read_vocab(vocab_file):
    vocab2id = {}
    id2vocab = {}
    for index,line in enumerate([line.strip() for line in open(vocab_file,"r",encoding='utf-8').readlines()]):
        vocab2id[line] = index
        id2vocab[index] = line
    return vocab2id, id2vocab

# print(read_vocab("./data/tags.txt"))



def tokenize(filename,vocab2id,tag2id):
    contents = []
    labels = []
    content = []
    label = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in [elem.strip() for elem in fr.readlines()][:500000]:
            try:
                if line != "end":
                    w,t = line.split()
                    content.append(vocab2id.get(w,vocab2id['<UNK>']))
                    label.append(tag2id.get(t,tag2id['<UNK>']))
                else:
                    if content and label:
                        contents.append(content)
                        labels.append(label)
                    content = []
                    label = []
            except Exception as e:
                content = []
                label = []
    contents = tf.keras.preprocessing.sequence.pad_sequences(contents, padding='post')
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, padding='post')
    return contents,labels,len(contents[0])






tag_check = {
    "I":["B","I"],
    "E":["B","I"]
}


def check_label(front_label,follow_label):
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
    print(entities)
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
    text = ['国','家','发','展','计','划','委','员','会','副','主','任','王','春','正']
    tags =  ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'E-ORG', 'O', 'O', 'O', 'B-PER', 'I-PER', 'E-PER']
    entities_result= format_result(text,tags)
    print(json.dumps(entities_result, indent=4, ensure_ascii=False))

