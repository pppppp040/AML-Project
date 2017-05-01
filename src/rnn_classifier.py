import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.engine.topology import Merge
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
from get_all_tags import get_all_tags

all_tags = get_all_tags()

data = pd.read_csv('../ProcessedData/biology_processed.csv', sep=',')
tag_list = data.tags.values.astype(str)

tk = text.Tokenizer(nb_words=200000)

max_len = 100
tk.fit_on_texts(list(data.title.values.astype(str))))
x1 = tk.texts_to_sequences(data.title.values)
x1 = sequence.pad_sequences(x1, maxlen=max_len)

word_index = tk.word_index

ytrain_enc = np_utils.to_categorical(y)

embeddings_index = {}
f = open('data/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

max_features = 200000
filter_length = 5
nb_filter = 64
pool_length = 4


model = Sequential()
model.add(Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2))
model.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))



model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

model.fit([x1], y=y, batch_size=384, nb_epoch=1,
                 verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])

