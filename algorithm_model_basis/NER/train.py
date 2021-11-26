# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/5/21 14:17
# 文件     ：train.py
# IDE     : PyCharm
from utils import tokenize, build_vocab, read_vocab, mkdir, load_embedding_matrix
import tensorflow as tf
from model import NerModel, NerModelWord
import tensorflow_addons as tf_ad
import os
import numpy as np
from args_help import args
from my_log import logger
from eval import conlleval
from gensim.models import Word2Vec, FastText

mkdir('eval')
if not (os.path.exists(args.vocab_file) and os.path.exists(args.tag_file)):
    logger.info("building vocab file")
    build_vocab([args.train_path], args.vocab_file, args.tag_file)
else:
    logger.info("vocab file exits!!")
vocab2id, id2vocab = read_vocab(args.vocab_file)
tag2id, id2tag = read_vocab(args.tag_file)

model_fasttext = FastText.load(args.word_per_path)
id2word = {i + 2: j for i, j in enumerate(model_fasttext.wv.index_to_key)}
id2word[0] = '<PAD>'
id2word[1] = '<UNK>'
word2id = {j: i for i, j in id2word.items()}
embedding_matrix = model_fasttext.syn1neg
embedding_size = embedding_matrix.shape[1]  # embedding size
embedding_matrix = np.concatenate([np.zeros((1, embedding_size)), embedding_matrix])  # 第0行，padding向量
embedding_matrix = np.concatenate([np.random.random((1, embedding_size)), embedding_matrix])
with open(r'data/vocab_word.txt', 'w', encoding='utf-8') as fp:
    ws = list(word2id.keys())
    for w in ws:
        fp.write(w)
        fp.write('\n')

if args.mix_word_char:
    text_sequences, label_sequences, _, words = tokenize(args.train_path, vocab2id, tag2id, word2id)
else:
    text_sequences, label_sequences, _ = tokenize(args.train_path, vocab2id, tag2id)


def train():
    train_dataset = tf.data.Dataset.from_tensor_slices((text_sequences, label_sequences))
    train_dataset = train_dataset.shuffle(len(text_sequences)).batch(args.batch_size, drop_remainder=True)

    logger.info("hidden_num:{}, vocab_size:{}, label_size:{}".format(args.hidden_num, len(vocab2id), len(tag2id)))
    model = NerModel(hidden_num=args.hidden_num, vocab_size=len(vocab2id), label_size=len(tag2id),
                     embedding_size=args.embedding_size)
    #     model._set_inputs(tf.random.normal([[1, 2, 3]], dtype=tf.int64))  # add this line
    optimizer = tf.keras.optimizers.Adam(args.lr)

    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt.restore(tf.train.latest_checkpoint(args.output_dir))
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              args.output_dir,
                                              checkpoint_name='model.ckpt',
                                              max_to_keep=3)

    def train_one_step(text_batch, labels_batch):
        with tf.GradientTape() as tape:
            logits, text_lens, log_likelihood, _ = model(text_batch, labels_batch)
            loss = - tf.reduce_mean(log_likelihood)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, logits, text_lens

    def get_acc_one_step(text_batch, logits, text_lens, labels_batch):
        paths = []
        model_predict = []
        for chars, logit, text_len, labels in zip(text_batch, logits, text_lens, labels_batch):
            viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
            for j in range(text_len):
                model_predict.append([id2vocab[chars[j].numpy()], id2tag[viterbi_path[j]], id2tag[labels[j].numpy()]])
        metrics = conlleval(model_predict, 'eval/label.txt', 'eval/metric.txt')
        return metrics

    best_f1 = 0
    step = 0
    sess = tf.compat.v1.Session()
    for epoch in range(args.epoch):
        for _, (text_batch, labels_batch) in enumerate(train_dataset):
            step = step + 1
            loss, logits, text_lens = train_one_step(text_batch, labels_batch)
            if step % 20 == 0:
                metrics = get_acc_one_step(text_batch, logits, text_lens, labels_batch)
                logger.info('===============================================================================')
                logger.info('epoch %d, step %d, loss %.4f' % (epoch, step, loss))
                logger.info('\n')
                for _ in metrics:
                    logger.info(_)
                logger.info('-------------------------------------------------------------------------------')
                f1 = float(metrics[1].split(':')[-1])
                if f1 > best_f1:
                    best_f1 = f1
                    ckpt_manager.save()
                    if args.PbFlag == True:
                        print(args.PbFlag)
                        logger.info("******************** pb save *************************")
                        tf.saved_model.save(model, args.savemode_dir)
                    logger.info("************************** model saved ************************************")

    logger.info("finished")


def train_word():
    train_dataset = tf.data.Dataset.from_tensor_slices((text_sequences, words, label_sequences))
    train_dataset = train_dataset.shuffle(len(text_sequences)).batch(args.batch_size, drop_remainder=False)

    logger.info(
        "hidden_num:{}, char_size:{}, label_size:{}, word_size:{}".format(args.hidden_num, len(vocab2id), len(tag2id),
                                                                          len(word2id)))

    model = NerModelWord(hidden_num=args.hidden_num, char_size=len(vocab2id), label_size=len(tag2id),
                         embedding_size=args.embedding_size, word_size=len(word2id), embedding_matrix=embedding_matrix)
    optimizer = tf.keras.optimizers.Adam(args.lr)

    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt.restore(tf.train.latest_checkpoint(args.output_dir))
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              args.output_dir,
                                              checkpoint_name='model.ckpt',
                                              max_to_keep=3)

    def train_one_step(char_batch, word_batch, labels_batch):
        with tf.GradientTape() as tape:
            logits, text_lens, log_likelihood, _ = model(char_batch, word_batch, labels_batch)
            loss = - tf.reduce_mean(log_likelihood)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, logits, text_lens

    def get_acc_one_step(char_batch, logits, text_lens, labels_batch):
        paths = []
        model_predict = []
        for chars, logit, text_len, labels in zip(char_batch, logits, text_lens, labels_batch):
            viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
            for j in range(text_len):
                model_predict.append([id2vocab[chars[j].numpy()], id2tag[viterbi_path[j]], id2tag[labels[j].numpy()]])
        metrics = conlleval(model_predict, 'eval/label.txt', 'eval/metric.txt')
        return metrics

    best_f1 = 0
    step = 0
    sess = tf.compat.v1.Session()
    for epoch in range(args.epoch):
        for _, (char_batch, word_batch, labels_batch) in enumerate(train_dataset):
            step = step + 1
            loss, logits, text_lens = train_one_step(char_batch, word_batch, labels_batch)
            if step % 100 == 0:
                metrics = get_acc_one_step(char_batch, logits, text_lens, labels_batch)
                logger.info('===============================================================================')
                logger.info('epoch %d, step %d, loss %.4f' % (epoch, step, loss))
                logger.info('\n')
                for _ in metrics:
                    logger.info(_)
                logger.info('-------------------------------------------------------------------------------')
                f1 = float(metrics[1].split(':')[-1])
                if f1 > best_f1:
                    best_f1 = f1
                    ckpt_manager.save()
                    if args.PbFlag == True:
                        print(args.PbFlag)
                        logger.info("******************** pb save *************************")
                        tf.saved_model.save(model, args.savemode_dir)
                    logger.info("************************** model saved ************************************")

    logger.info("finished")


if __name__ == '__main__':
    if args.mix_word_char:
        train_word()
    else:
        train()

