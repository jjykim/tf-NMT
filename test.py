import tensorflow as tf
import math

from text_util import build_vocab, read_file
from model import Seq2SeqModel


class DemoConfig:

    #file
    sc_file='./data/small_en.csv'
    tg_file='./data/small_vi.csv'
    sc_ts_file='./data/small_en.csv'
    tg_ts_file='./data/small_vi.csv'
    batch_size = 10
    data_line_size = 40
    enc_sentence_length = 150
    dec_sentence_length = 150


    # Model
    hidden_size = 50
    enc_emb_size = 50
    dec_emb_size = 50
    attn_size = 50
    cell = tf.contrib.rnn.BasicLSTMCell

    # Training
    optimizer = tf.train.AdamOptimizer
    n_epoch = 2
    learning_rate = 0.001

    # Tokens
    start_token = 0 # GO
    end_token = 1 # PAD

    save_dir = './save_dir/'

config = DemoConfig()

#make vocab
config.enc_vocab, config.enc_reverse_vocab, config.enc_vocab_size = build_vocab(DemoConfig.sc_file)
config.dec_vocab, config.dec_reverse_vocab, config.dec_vocab_size = build_vocab(DemoConfig.tg_file, is_target=True)


with tf.Session() as sess:

    steps_per_epoch = math.ceil(config.data_line_size / config.batch_size)

    train_x = read_file(config.sc_file, config.batch_size)
    train_y = read_file(config.tg_file, config.batch_size)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    model = Seq2SeqModel(config, mode='training')
    model.build()
    print('Training model built!')
    model.train(sess, train_x, train_y, steps_per_epoch, save_path=model.save_dir+f'epoch_{model.n_epoch}')

    coord.request_stop()
    coord.join(threads)
    sess.close

tf.reset_default_graph()
with tf.Session() as sess:

    train_x = read_file(config.sc_ts_file, config.batch_size)
    train_y = read_file(config.tg_ts_file, config.batch_size)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    model = Seq2SeqModel(config, mode='inference')
    model.build()
    print('Inference model built!')
    model.inference(sess, train_x, train_y, save_path=model.save_dir+f'epoch_{model.n_epoch}')

    coord.request_stop()
    coord.join(threads)
    sess.close

