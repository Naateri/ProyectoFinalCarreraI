from __future__ import print_function
import numpy as np
import time 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Concatenate, GlobalMaxPool1D, GlobalAveragePooling1D, TimeDistributed
from tensorflow.keras.datasets import imdb
from tensorflow.keras.optimizers import RMSprop

#keras.utils.plot_model requires graphviz

max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 400
batch_size = 64
epochs = 45

print('Loading data...')
data = imdb.load_data(num_words=max_features)
#data = imdb.load_data()
(x_train, y_train), (x_test, y_test) = data
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

words = imdb.get_word_index()
words = {k:(v+3) for k,v in words.items()}
words["<PAD>"] = 0
words["<START>"] = 1
words["<UNK>"] = 2
words["<UNUSED>"] = 3

id_to_word = {value:key for key,value in words.items()}

get_sentence = lambda vector : ' '.join(id_to_word[num] for num in vector)

#print(get_sentence(x_train[0]), y_train[0])
#print(get_sentence(x_train[1]), y_train[1])

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

print("x shape: ", x_train[0].shape)

def transform(sentence, pad_len):
	vector = list()
	sentence = sentence.split()
	for word in sentence:	
		vector.append(words[word.lower()])
	# Saving as numpy array
	vector = np.array(vector) 
	# Saving as a matrix
	vector = np.array([vector])
	# Padding
	padded_vector = sequence.pad_sequences(vector, maxlen=pad_len)
	return padded_vector
	
example = transform("That guy sucks really bad buddy", maxlen)
print("example shape", example.shape)
#print("example in array shape", np.array([example]).shape)

net_optimizer = RMSprop(learning_rate=0.0001)

'''
model = Sequential()
#model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Embedding(max_features, 300, input_length=maxlen))
#model.add(Bidirectional(LSTM(64)))
model.add(Bidirectional(LSTM(16, dropout=0.3)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid',
                kernel_regularizer='l2'))
'''

inputs = keras.Input(shape=(maxlen,), name="InputLayer") # TODO: check shape
embedding = Embedding(max_features, 300, input_length=maxlen, 
		name="EmbeddingLayer") (inputs)
bilstm_layer = Bidirectional(LSTM(16, dropout=0.3, 
		return_sequences=True), name="BiLSTMLayer") (embedding)

gpooling_layer_max = GlobalMaxPool1D() (bilstm_layer)
gpooling_layer_avg = GlobalAveragePooling1D() (bilstm_layer)

concatenate_layer = Concatenate() ([gpooling_layer_max, gpooling_layer_avg])

outputs = Dense(1, activation='sigmoid', kernel_regularizer='l2', 
		name="OutputLayer") (concatenate_layer)

model = keras.Model(inputs=inputs, outputs=outputs)

# try using different optimizers and different optimizer configs
#model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=net_optimizer, loss='binary_crossentropy',
              metrics=['accuracy'])
              
#model.summary()
#keras.utils.plot_model(model, "Hammeed-BiLSTM.png")

print('Train...')
start = time.perf_counter()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=[x_test, y_test])
          
end = time.perf_counter()
print("Time to compile:", end-start)

model.save('test.h5')
#save_model(model, './saved_model')

predictions = model.predict(np.array(example))
print(predictions.shape)
print(predictions)
print(np.argmax(predictions, axis=1))

example = transform("He is an amazing driver loved it", maxlen)
predictions = model.predict(np.array(example))
print(predictions)
print(np.argmax(predictions, axis=1))

