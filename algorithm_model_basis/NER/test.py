# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/05/21 14:39
# 文件     ：test.py
# IDE     : PyCharm
import tensorflow as tf
from model import NerModel
from utils import tokenize,read_vocab,format_result
import tensorflow_addons as tf_ad
from args_help import args
import json


vocab2id, id2vocab = read_vocab(args.vocab_file)
tag2id, id2tag = read_vocab(args.tag_file)

test_sequences , label_test_seq , _= tokenize(args.test_path,vocab2id,tag2id)
test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, label_test_seq))
test_dataset = test_dataset.shuffle(len(test_sequences)).batch(args.batch_size, drop_remainder=True)



optimizer = tf.keras.optimizers.Adam(args.lr)
model = NerModel(hidden_num = args.hidden_num, vocab_size =len(vocab2id), label_size = len(tag2id), embedding_size = args.embedding_size)
# restore model
ckpt = tf.train.Checkpoint(optimizer=optimizer,model=model)
ckpt.restore(tf.train.latest_checkpoint(args.output_dir))
with open('result/test.txt','w',encoding='utf-8') as fp:
    for _, (text_batch, labels_batch) in enumerate(test_dataset):
        logits, text_lens = model.call(text_batch)
        paths = []
        i = 0
        for logit, text_len, labels in zip(logits, text_lens, labels_batch):
            if text_len == 0:
                continue
            viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)

            for j in range(text_len):
                print(id2vocab[text_batch[i][j].numpy()])
                fp.write(id2vocab[text_batch[i][j].numpy()]+' '+id2tag[viterbi_path[j]]+' '+id2tag[labels_batch[i][j].numpy()]+'\n')
#                 fp.write(id2vocab[text_batch[i][j].numpy()]+' '+id2tag[labels_batch[i][j].numpy()]+' '+id2tag[viterbi_path[j]]+'\n')
            fp.write('\n')
            i+=1
