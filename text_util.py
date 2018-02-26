import re
import tensorflow as tf

from collections import Counter


def tokenizer(sentence):
    tokens = re.findall(r"[\w]+|[^\s\w]", sentence)
    return tokens


def build_vocab(file, is_target=False, max_vocab_size=None):

    with open(file, "r") as f:
        lines = f.readlines()

        word_counter = Counter()
        vocab = dict()
        reverse_vocab = dict()

        for sentence in lines:
            tokens = tokenizer(sentence)
            word_counter.update(tokens)

        if max_vocab_size is None:
            max_vocab_size = len(word_counter)

        if is_target:
            vocab['_GO'] = 0
            vocab['_PAD'] = 1
            vocab_idx = 2
            for key, value in word_counter.most_common(max_vocab_size):
                vocab[key] = vocab_idx
                vocab_idx += 1
        else:
            vocab['_PAD'] = 0
            vocab_idx = 1
            for key, value in word_counter.most_common(max_vocab_size):
                vocab[key] = vocab_idx
                vocab_idx += 1

        for key, value in vocab.items():
            reverse_vocab[value] = key

        return vocab, reverse_vocab, max_vocab_size


def token2idx(word, vocab):
    return vocab[word]


def sent2idx(sent, vocab, max_sentence_length, is_target=False):
    tokens = tokenizer(sent)
    current_length = len(tokens)
    pad_length = max_sentence_length - current_length
    if is_target:
        return [0] + [token2idx(token, vocab) for token in tokens] + [1] * pad_length, current_length
    else:
        return [token2idx(token, vocab) for token in tokens] + [0] * pad_length, current_length


def idx2token(idx, reverse_vocab):
    return reverse_vocab[idx]


def idx2sent(indices, reverse_vocab):
    return " ".join([idx2token(idx, reverse_vocab) for idx in indices])


def read_file(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename], shuffle=False, name='filename_queue')

    reader = tf.TextLineReader()
    _, csv_row = reader.read(filename_queue)

    record_defaults = [['_PAD']]
    line = tf.decode_csv(csv_row, record_defaults=record_defaults, field_delim='\n')

    train_batch = tf.train.batch([line], batch_size=batch_size)

    return train_batch

