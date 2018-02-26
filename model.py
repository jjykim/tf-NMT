import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.python.layers.core import Dense
from tqdm import tqdm
from text_util import sent2idx, idx2sent


class Seq2SeqModel(object):
    def __init__(self, config, mode='training'):
        assert mode in ['training', 'inference']
        self.mode = mode

        self.enc_sentence_length = config.enc_sentence_length
        self.dec_sentence_length = config.dec_sentence_length
        self.enc_vocab_size = config.enc_vocab_size
        self.dec_vocab_size = config.dec_vocab_size

        self.enc_vocab = config.enc_vocab
        self.dec_vocab = config.dec_vocab
        self.enc_reverse_vocab = config.enc_reverse_vocab
        self.dec_reverse_vocab = config.dec_reverse_vocab

        # Model
        self.hidden_size = config.hidden_size
        self.enc_emb_size = config.enc_emb_size
        self.dec_emb_size = config.dec_emb_size
        self.attn_size = config.attn_size
        self.cell = config.cell

        # Training
        self.optimizer = config.optimizer
        self.n_epoch = config.n_epoch
        self.learning_rate = config.learning_rate

        # Tokens
        self.start_token = config.start_token
        self.end_token = config.end_token

        self.save_dir = config.save_dir


    def add_placeholders(self):
        self.enc_inputs = tf.placeholder(
            tf.int32,
            shape=[None, self.enc_sentence_length],
            name='input_sentences')

        self.enc_sequence_length = tf.placeholder(
            tf.int32,
            shape=[None,],
            name='input_sequence_length')

        if self.mode == 'training':
            self.dec_inputs = tf.placeholder(
                tf.int32,
                shape=[None, self.dec_sentence_length+1],
                name='target_sentences')

            self.dec_sequence_length = tf.placeholder(
                tf.int32,
                shape=[None,],
                name='target_sequence_length')

    def add_encoder(self):
        with tf.variable_scope('Encoder') as scope:
            with tf.device('/cpu:0'):
                self.enc_Wemb = tf.get_variable('embedding',
                    initializer=tf.random_uniform([self.enc_vocab_size+1, self.enc_emb_size]),
                    dtype=tf.float32)

            enc_emb_inputs = tf.nn.embedding_lookup(
                self.enc_Wemb, self.enc_inputs, name='emb_inputs')
            enc_cell = self.cell(self.hidden_size)

            self.enc_outputs, self.enc_last_state = tf.nn.dynamic_rnn(
                cell=enc_cell,
                inputs=enc_emb_inputs,
                sequence_length=self.enc_sequence_length,
                time_major=False,
                dtype=tf.float32)

            # forward_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            # backward_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            #
            # bi_outputs, self.enc_last_state = tf.nn.bidirectional_dynamic_rnn(
            #     cell_fw=forward_cell, cell_bw=backward_cell, inputs=enc_emb_inputs, dtype=tf.float32,
            #     sequence_length=self.enc_sequence_length, time_major=False)
            # self.enc_outputs = tf.concat(bi_outputs, -1)


    def add_decoder(self):
        with tf.variable_scope('Decoder') as scope:
            with tf.device('/cpu:0'):
                self.dec_Wemb = tf.get_variable('embedding',
                    initializer=tf.random_uniform([self.dec_vocab_size+2, self.dec_emb_size]),
                    dtype=tf.float32)

            batch_size = tf.shape(self.enc_inputs)[0]

            dec_cell = self.cell(self.hidden_size)

            attn_mech = tf.contrib.seq2seq.LuongAttention(
                num_units=self.attn_size,
                memory=self.enc_outputs,
                memory_sequence_length=self.enc_sequence_length,
                name='LuongAttention')

            dec_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=dec_cell,
                attention_mechanism=attn_mech,
                attention_layer_size=self.attn_size,
                name='Attention_Wrapper')

            initial_state = dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size).clone(
          cell_state=self.enc_last_state)

            output_layer = Dense(self.dec_vocab_size+2, name='output_projection')

            if self.mode == 'training':

                max_dec_len = tf.reduce_max(self.dec_sequence_length+1, name='max_dec_len')

                dec_emb_inputs = tf.nn.embedding_lookup(
                    self.dec_Wemb, self.dec_inputs, name='emb_inputs')

                training_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=dec_emb_inputs,
                    sequence_length=self.dec_sequence_length+1,
                    time_major=False,
                    name='training_helper')

                training_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=dec_cell,
                    helper=training_helper,
                    initial_state=initial_state,
                    output_layer=output_layer)

                train_dec_outputs, train_dec_last_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_dec_len)

                logits = tf.identity(train_dec_outputs.rnn_output, name='logits')

                targets = tf.slice(self.dec_inputs, [0, 0], [-1, max_dec_len], 'targets')

                masks = tf.sequence_mask(self.dec_sequence_length+1, max_dec_len, dtype=tf.float32, name='masks')

                self.batch_loss = tf.contrib.seq2seq.sequence_loss(
                    logits=logits,
                    targets=targets,
                    weights=masks,
                    name='batch_loss')

                self.valid_predictions = tf.identity(train_dec_outputs.sample_id, name='valid_preds')

            elif self.mode == 'inference':

                start_tokens = tf.tile(tf.constant([self.start_token], dtype=tf.int32), [batch_size], name='start_tokens')

                inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=self.dec_Wemb,
                    start_tokens=start_tokens,
                    end_token=self.end_token)

                inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=dec_cell,
                    helper=inference_helper,
                    initial_state=initial_state,
                    output_layer=output_layer)

                infer_dec_outputs, infer_dec_last_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    inference_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=self.dec_sentence_length)

                self.predictions = tf.identity(infer_dec_outputs.sample_id, name='predictions')


    def add_training_op(self):
        self.training_op = self.optimizer(self.learning_rate, name='training_op').minimize(self.batch_loss)

    def build(self):
        self.add_placeholders()
        self.add_encoder()
        self.add_decoder()

    def save(self, sess, var_list=None, save_path=None):
        print('Saving model ')
        if hasattr(self, 'training_variables'):
            var_list = self.training_variables
        saver = tf.train.Saver(var_list)
        saver.save(sess, save_path, write_meta_graph=False)

    def restore(self, sess, var_list=None, save_path=None):
        if hasattr(self, 'training_variables'):
            var_list = self.training_variables
        self.restorer = tf.train.Saver(var_list)
        self.restorer.restore(sess, save_path)
        print('Restore!')

    def train(self, sess, train_x, train_y, steps_per_epoch, save_path):

        self.add_training_op()

        sess.run(tf.global_variables_initializer())

        loss_history = []
        for epoch in tqdm(range(self.n_epoch)):
            all_preds = []
            epoch_loss = 0

            for i in range(steps_per_epoch):
                single_input_batch, single_target_batch = sess.run([train_x, train_y])

                input_batch_tokens = []
                target_batch_tokens = []
                enc_sentence_lengths = []
                dec_sentence_lengths = []

                for sc_line, tg_line in zip(single_input_batch, single_target_batch):
                    tokens, sent_len = sent2idx(sc_line[0].decode('utf-8'), vocab=self.enc_vocab, max_sentence_length=self.enc_sentence_length)
                    input_batch_tokens.append(tokens)
                    enc_sentence_lengths.append(sent_len)

                    tokens, sent_len = sent2idx(tg_line[0].decode('utf-8'),
                                 vocab=self.dec_vocab,
                                 max_sentence_length=self.dec_sentence_length,
                                 is_target=True)
                    target_batch_tokens.append(tokens)
                    dec_sentence_lengths.append(sent_len)

                batch_pred, batch_loss, _ = sess.run(
                    [self.valid_predictions, self.batch_loss, self.training_op],
                    feed_dict={
                        self.enc_inputs: input_batch_tokens,
                        self.enc_sequence_length: enc_sentence_lengths,
                        self.dec_inputs: target_batch_tokens,
                        self.dec_sequence_length: dec_sentence_lengths,
                    })

                epoch_loss += batch_loss
                all_preds.append(batch_pred)

            loss_history.append(epoch_loss)

            if epoch % 50 == 0:
                for sc_line, tg_line, line_pred in zip(single_input_batch, single_target_batch, batch_pred):
                    print('\tInput:', sc_line[0].decode('utf-8'))
                    print('\tPrediction:', idx2sent(line_pred, reverse_vocab=self.dec_reverse_vocab))
                    print('\tTarget:', tg_line[0].decode('utf-8'))
                print(f'\tepoch loss: {epoch_loss:.2f}\n')

        if save_path:
            self.save(sess, save_path=save_path)

        self.show_loss_graph(loss_history)
        return batch_loss, batch_pred

    def inference(self, sess, train_x, train_y, save_path):

        self.restore(sess, save_path=save_path)

        batch_preds = []
        batch_tokens = []
        batch_sent_lens = []

        single_input_batch, single_target_batch = sess.run([train_x, train_y])

        for input_sent in single_input_batch:
            tokens, sent_len = sent2idx(input_sent[0].decode('utf-8'), vocab=self.enc_vocab, max_sentence_length=self.enc_sentence_length)
            batch_tokens.append(tokens)
            batch_sent_lens.append(sent_len)

        batch_preds = sess.run(
            self.predictions,
            feed_dict={
                self.enc_inputs: batch_tokens,
                self.enc_sequence_length: batch_sent_lens,
            })

        for input_sent, target_sent, pred in zip(single_input_batch, single_target_batch, batch_preds):
            print('Input:', input_sent[0].decode('utf-8'))
            print('Prediction:', idx2sent(pred, reverse_vocab=self.dec_reverse_vocab))
            print('Target:', target_sent[0].decode('utf-8'), '\n')

    def show_loss_graph(self, loss_history):
        plt.figure(figsize=(20, 10))
        plt.scatter(range(self.n_epoch), loss_history)
        plt.title('Learning Curve')
        plt.xlabel('Global step')
        plt.ylabel('Loss')
        plt.show()
