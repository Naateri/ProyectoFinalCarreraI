from __future__ import print_function
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.datasets import imdb
from tensorflow.keras.optimizers import RMSprop

max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 400
batch_size = 64

print('Loading data...')
data = imdb.load_data(num_words=max_features)
#data = imdb.load_data()
(x_train, y_train), (x_test, y_test) = data
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

net_optimizer = RMSprop(learning_rate=0.0001)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
#model.add(Bidirectional(LSTM(64)))
model.add(Bidirectional(LSTM(16)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid',
                kernel_regularizer='l2'))

# try using different optimizers and different optimizer configs
#model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=net_optimizer, loss='binary_crossentropy',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=45,
          validation_data=[x_test, y_test])
