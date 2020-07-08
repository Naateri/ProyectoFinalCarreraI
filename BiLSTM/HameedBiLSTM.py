from __future__ import print_function
import numpy as np
import time
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Concatenate, GlobalMaxPool1D, GlobalAveragePooling1D, TimeDistributed
from tensorflow.keras.optimizers import RMSprop

class HameedBiLSTM:

	model = None
	embeddings_index = {}

	def load_glove(self):
		print('Building Glove data')
		GLOVE_DIR = 'datasets/glove/'
		f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			self.embeddings_index[word] = coefs
		f.close()

		print('Glove: Found %s word vectors.' % len(self.embeddings_index))

	def __init__(self, maxlen, max_features, /, use_glove=False, word_index=None):
		inputs = keras.Input(shape=(maxlen,), name="InputLayer")
		if use_glove:
			self.load_glove()
			embedding_matrix = np.zeros((len(word_index)+1, 300))
			for word, i in word_index.items():
				embedding_vector = self.embeddings_index.get(word)
				if embedding_vector is not None:
					# if not found: embedding_matrix will be all zeros
					embedding_matrix[i] = embedding_vector
			
			embedding = Embedding(len(word_index) + 1, 300, input_length=maxlen,
				weights=[embedding_matrix], trainable=True) (inputs)
		else:
			embedding = Embedding(max_features, 300, input_length=maxlen, trainable=True,
				name="EmbeddingLayer") (inputs)
		bilstm_layer = Bidirectional(LSTM(16, dropout=0.3, 
			return_sequences=True), name="BiLSTMLayer") (embedding)

		gpooling_layer_max = GlobalMaxPool1D() (bilstm_layer)
		gpooling_layer_avg = GlobalAveragePooling1D() (bilstm_layer)

		concatenate_layer = Concatenate() ([gpooling_layer_max, gpooling_layer_avg])

		outputs = Dense(1, activation='sigmoid', kernel_regularizer='l2', 
		name="OutputLayer") (concatenate_layer)

		self.model = keras.Model(inputs=inputs, outputs=outputs)

		self.network_optimizer = RMSprop(learning_rate=0.0001)
		# try using different optimizers and different optimizer configs
		#model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
		self.model.compile(optimizer=self.network_optimizer, loss='binary_crossentropy',
					metrics=['accuracy'])

	def train(self, x_train, x_test, y_train, y_test, /, batch_size=64, epochs=45, 
		save_model = False, save_file = 'models/bilstm_model'):
		print('Train...')
		start = time.perf_counter()

		self.model.fit(x_train, y_train,
				batch_size=batch_size,
				epochs=epochs,
				validation_data=[x_test, y_test])
				
		end = time.perf_counter()
		print("Time to train:", end-start)
		self.time_to_train = end-start

		if save_model:
			self.model.save(save_file)
	
	def load(self, filename):
		model = load_model(filename)
		return model

#keras.utils.plot_model requires graphviz
