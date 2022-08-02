from utils import tokenize, build_vocab, read_vocab, mkdir, read_wordvocab
import tensorflow as tf
from model import NerModel
import tensorflow_addons as tf_ad
import os
import numpy as np
from args_help import args
from my_log import logger
from eval import conlleval
import tensorflow.keras as keras
from tf2crf import CRF, ModelWithCRFLoss

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

mkdir('eval')
if not (os.path.exists(args.vocab_file) and os.path.exists(args.tag_file)):
    logger.info("building vocab file")
    build_vocab([args.train_path], args.vocab_file, args.tag_file)
else:
    logger.info("vocab file exits!!")
vocab2id, id2vocab = read_vocab(args.vocab_file)
tag2id, id2tag = read_vocab(args.tag_file)
embeddings_matrix, word2idx = read_wordvocab(args.fastvec_dir)
text_sequences, words_sequences, label_sequences, _ = tokenize(args.train_path, vocab2id, tag2id, word2idx, 1)


def train():
    train_dataset = tf.data.Dataset.from_tensor_slices((text_sequences, words_sequences, label_sequences))
    train_dataset = train_dataset.shuffle(len(text_sequences)).batch(args.batch_size, drop_remainder=True)
    logger.info("hidden_num:{}, vocab_size:{}, label_size:{}".format(args.hidden_num, len(vocab2id), len(tag2id)))
    model = NerModel(hidden_num=args.hidden_num, vocab_size=len(vocab2id), label_size=len(tag2id),
                     embedding_size=args.embedding_size, dropout_rate=args.dropout_rate,
                     embeddings_matrix=embeddings_matrix, sentence_size=len(word2idx))
    optimizer = tf.keras.optimizers.Adam(args.lr)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    #     ckpt.restore(tf.train.latest_checkpoint(args.output_dir))
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              args.output_dir,
                                              checkpoint_name='model.ckpt',
                                              max_to_keep=3)

    def train_one_step(text_batch, word_batch, labels_batch):
        with tf.GradientTape() as tape:
            logits, text_lens, log_likelihood, _ = model(text_batch, word_batch, labels_batch)
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
        for _, (text_batch, word_batch, labels_batch) in enumerate(train_dataset):
            step = step + 1
            loss, logits, text_lens = train_one_step(text_batch, word_batch, labels_batch)
        metrics = get_acc_one_step(text_batch, logits, text_lens, labels_batch)
        logger.info('===============================================================================')
        logger.info('epoch %d, step %d, loss %.4f' % (epoch, step, loss))
        logger.info('\n')
        for _ in metrics:
            logger.info(_)
        logger.info('-------------------------------------------------------------------------------')
    ckpt_manager.save()
    if args.PbFlag == True:
        print(args.PbFlag)
        logger.info("******************** pb save *************************")
        tf.saved_model.save(model, args.savemode_dir, signatures={"serving_default": model.call})
    logger.info("************************** model saved ************************************")

    logger.info("finished")


if __name__ == '__main__':
    train()

