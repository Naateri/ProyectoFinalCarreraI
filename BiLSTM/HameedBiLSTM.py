from __future__ import print_function
import numpy as np
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Concatenate, GlobalMaxPool1D, GlobalAveragePooling1D, TimeDistributed
from tensorflow.keras.optimizers import RMSprop

class HameedBiLSTM:

	model = None

	def __init__(self, maxlen, max_features):
		inputs = keras.Input(shape=(maxlen,), name="InputLayer")
		embedding = Embedding(max_features, 300, input_length=maxlen, 
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
