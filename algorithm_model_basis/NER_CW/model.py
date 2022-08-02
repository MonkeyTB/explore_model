
import tensorflow as tf
import tensorflow_addons as tf_ad

class NerModel(tf.keras.Model):
    def __init__(self, hidden_num, vocab_size, label_size, embedding_size, dropout_rate, embeddings_matrix,sentence_size):
        super(NerModel, self).__init__()
        self.num_hidden = hidden_num
        self.vocab_size = vocab_size
        self.label_size = label_size

        self.embeddingWord = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.embeddingSentenct = tf.keras.layers.Embedding(sentence_size,
                                                           embedding_size,
                                                          weights = [embeddings_matrix],
                                                          trainable=False)
        self.add = tf.keras.layers.Add
        self.concat = tf.keras.layers.concatenate
        self.biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_num, return_sequences=True))
        self.dense = tf.keras.layers.Dense(label_size)

        self.transition_params = tf.Variable(tf.random.uniform(shape=(label_size, label_size)))
#         self.dropout = tf.keras.layers.Dropout(dropout_rate)


    @tf.function(input_signature=(tf.TensorSpec([None, None], tf.int32), tf.TensorSpec([None, None], tf.int32), tf.TensorSpec([None, None], tf.int32)))
    def call(self, text, sentence, labels=None):
        # -1 change 0
        inputWord = self.embeddingWord(text)
        inputSentene = self.embeddingSentenct(sentence)
        inputs = self.concat([inputWord, inputSentene], axis = -1)
#         inputs = self.add()([inputWord, inputSentene])
        inputs = self.biLSTM(inputs)
        logits = self.dense(inputs)
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(text, 0), dtype=tf.int32), axis=-1)
        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(logits,
                                                                                   label_sequences,
                                                                                   text_lens,
                                                                                   transition_params=self.transition_params)
            self.crf_dec = tf_ad.text.crf_decode(logits,self.transition_params,text_lens)
            return logits, text_lens, log_likelihood, self.crf_dec[0]
        else:
            return logits, text_lens, self.crf_dec[0]
