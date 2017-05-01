import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
from get_all_tags import get_all_tags
import sys
import re

all_tags = get_all_tags()

src_domain = ['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']

tk_titles = text.Tokenizer(num_words = 200000)
tk_tags = text.Tokenizer(num_words = 200000)

titles_max_len = 0

def read_data(csv_file):

    data = pd.read_csv(csv_file, sep = ',')
    titles = data.title.values.astype(str)
    lst_tags_strings = data.tags.values.astype(str)
    tags_str = []
    for tag_str in lst_tags_strings:
        tag_str = re.sub(r"[^A-Za-z0-9]", " ", tag_str)
        # tag_str_words = tag_str.strip().split()
        tags_str.append(tag_str)

    return (titles, tags_str)

def get_one_hot_vectors(tags_seqs, tags_index):

    print 'type(tags_seqs): ', type(tags_seqs)

    print 'type(tags_index): ', type(tags_index)

    tags_seqs_onehot = np.zeros((len(tags_seqs), len(tags_index) + 1))

    for i, tag_seq in enumerate(tags_seqs):
        for idx in tag_seq:
            if idx == 710:
                print tag_seq
            tags_seqs_onehot[i][idx] = 1

    return tags_seqs_onehot

def tokenize(titles, tags_str):

    tk_titles.fit_on_texts(titles)

    tk_tags.fit_on_texts(tags_str)

    titles_seqs = tk_titles.texts_to_sequences(titles)

    print 'type(titles_seqs): ', type(titles_seqs)

    titles_max_len = 0

    for titles_seq in titles_seqs:
        titles_max_len = max(titles_max_len, len(titles_seq))

    print 'titles_max_len: ', titles_max_len

    titles_seqs_padded = sequence.pad_sequences(titles_seqs, maxlen = titles_max_len)

    tags_seqs = tk_tags.texts_to_sequences(tags_str)

    print 'type(tags_seqs): ', type(tags_seqs)

    tags_index = tk_tags.word_index

    tags_seqs_onehot = get_one_hot_vectors(tags_seqs, tags_index)

    return titles_seqs_padded, tags_seqs_onehot

def load_embeddings():

    embeddings_index = {}
    f = open('../Data/glove.840B.300d.txt')
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')
        embeddings_index[word] = coefs
        if len(embeddings_index) > 100:
            break

    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(tk_titles.word_index) + 1, 300))
    for word, i in tqdm(tk_titles.word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def build_and_train(titles_seqs_padded, tags_seqs_onehot, embedding_matrix):

    model = Sequential()

    model.add(Embedding(len(tk_titles.word_index) + 1,
                         300,
                         weights = [embedding_matrix],
                         input_length = titles_max_len,
                         trainable = False))

    model.add(LSTM(300, dropout_W = 0.2, dropout_U = 0.2))

    model.add(BatchNormalization())

    model.add(Dense(300))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(len(tk_tags.word_index) + 1))

    model.add(Activation('sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    checkpoint = ModelCheckpoint('weights.h5', monitor = 'val_acc', save_best_only = True, verbose = 2)

    model.fit(titles_seqs_padded, y = tags_seqs_onehot, batch_size = 256, nb_epoch = 1,
                     verbose = 1, validation_split = 0.1, shuffle = True, callbacks = [checkpoint])

def main():

    root_dir = '../ProcessedData/'

    domain_id = int(sys.argv[1])

    titles, tags_str = read_data(root_dir + src_domain[domain_id] + '_processed.csv')  # TODO:

    titles_seqs_padded, tags_seqs_onehot = tokenize(titles, tags_str)

    embedding_matrix = load_embeddings()

    build_and_train(titles_seqs_padded, tags_seqs_onehot, embedding_matrix)


if __name__ == '__main__':
    main()
