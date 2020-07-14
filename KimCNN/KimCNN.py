# Code based on:
# https://kaggle.com/hamishdickson/cnn-for-sentence-classification-by-yoon-kim
from __future__ import print_function
import numpy as np
import time
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv2D, MaxPool2D, Reshape, Concatenate, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.constraints import max_norm

class KimCNN:

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

		num_filters = 100

		inputs = keras.Input(shape=(maxlen,), name="InputLayer") # Input

		if use_glove:
			self.load_glove()
			embedding_matrix = np.zeros((len(word_index)+1, 300))
			for word, i in word_index.items():
				embedding_vector = self.embeddings_index.get(word)
				if embedding_vector is not None:
					# if not found: embedding_matrix will be all zeros
					embedding_matrix[i] = embedding_vector
			
			embedding_layer = Embedding(len(word_index) + 1, 300, input_length=maxlen,
				weights=[embedding_matrix], trainable=True) (inputs)
		else:
			embedding_layer = Embedding(max_features, 300, input_length=maxlen, 
				trainable=True, name="EmbeddingLayer")(inputs) # Embedding layer

		# Regularizer might be the cause of bad results for 10 or less epochs
		'''
		reshape = Reshape((maxlen, 300, 1)) (embedding_layer)

		conv_3 = Conv2D(num_filters, kernel_size=(3, 300), activation='relu',
			kernel_initializer='normal') (reshape)
			#kernel_regularizer=keras.regularizers.l2(3), kernel_initializer='normal') (reshape)
			#kernel_regularizer='l2', kernel_initializer='normal', kernel_constraint=max_norm(3)) (reshape)
		
		conv_4 = Conv2D(num_filters, kernel_size=(4,300), activation='relu',
			kernel_initializer='normal') (reshape)
			#kernel_regularizer=keras.regularizers.l2(3), kernel_initializer='normal') (reshape)
			#kernel_regularizer='l2', kernel_initializer='normal', kernel_constraint=max_norm(3)) (reshape)
		
		conv_5 = Conv2D(num_filters, kernel_size=(5,300), activation='relu',
			kernel_initializer='normal') (reshape)
			#kernel_regularizer=keras.regularizers.l2(3), kernel_initializer='normal') (reshape)
			#kernel_regularizer='l2', kernel_initializer='normal', kernel_constraint=max_norm(3)) (reshape)

		maxpool_3 = MaxPool2D(pool_size=(maxlen - 3 + 1, 1), strides=(1,1),
			padding='valid') (conv_3)
		#maxpool_3 = MaxPool2D(pool_size=(maxlen - 3 + 1, 1)) (conv_3)

		maxpool_4 = MaxPool2D(pool_size=(maxlen - 4 + 1, 1), strides=(1,1),
			padding='valid') (conv_4)
		#maxpool_4 = MaxPool2D(pool_size=(maxlen - 4 + 1, 1)) (conv_4)
		
		maxpool_5 = MaxPool2D(pool_size=(maxlen - 5 + 1, 1), strides=(1,1),
			padding='valid') (conv_5)
		#maxpool_5 = MaxPool2D(pool_size=(maxlen - 5 + 1, 1)) (conv_5)

		concatenate_layer = Concatenate(axis=1)([maxpool_3, maxpool_4, maxpool_5])
		#concatenate_layer = Concatenate()([maxpool_3, maxpool_4, maxpool_5])
		'''
		
		conv_3 = Conv1D(num_filters, 3, activation='relu', padding='same') (embedding_layer)
			#kernel_regularizer=keras.regularizers.l2(3)) (embedding_layer)
		conv_4 = Conv1D(num_filters, 4, activation='relu', padding='same') (embedding_layer)
			#kernel_regularizer=keras.regularizers.l2(3)) (embedding_layer)
		conv_5 = Conv1D(num_filters, 5, activation='relu', padding='same') (embedding_layer)
			#kernel_regularizer=keras.regularizers.l2(3)) (embedding_layer)

		maxpool_3 = MaxPooling1D() (conv_3)
		maxpool_4 = MaxPooling1D() (conv_4)
		maxpool_5 = MaxPooling1D() (conv_5)

		concatenate_layer = Concatenate(axis=-1)([maxpool_3, maxpool_4, maxpool_5])
		
		
		flatten_layer = Flatten()(concatenate_layer)

		new_dense = Dense(num_filters, activation='relu') (flatten_layer)
		dropout_layer = Dropout(0.5)(new_dense)

		#dropout_layer = Dropout(0.5)(flatten_layer)

		outputs = Dense(1, activation='sigmoid') (dropout_layer)
			#kernel_regularizer='l2') (dropout_layer)

		self.model = keras.Model(inputs=inputs, outputs=outputs)

		#self.model.summary()

		#self.network_optimizer = RMSprop(learning_rate=0.0001)

		self.model.compile(optimizer='adam', loss='binary_crossentropy',
					metrics=['accuracy']) # try adam or adadelta, BEWARE: adadelta is too slow

	def train(self, x_train, x_test, y_train, y_test, /, batch_size=50, epochs=32, 
		save_model = False, save_file = 'models/kimcnn_model'):
		print('Train...')
		start = time.perf_counter()

		self.model.fit(x_train, y_train,
				batch_size=batch_size,
				epochs=epochs,
				validation_data=[x_test, y_test])
				#verbose=1,
				#validation_split=0.2)
				
		end = time.perf_counter()
		print("Time to train:", end-start)
		self.time_to_train = end-start

		if save_model:
			self.model.save(save_file)
	
	def load(self, filename):
		model = load_model(filename)
		return model

#keras.utils.plot_model requires graphviz

# Variar parámetros para ver distintos comportamientos
# Guardar valores en algún archivo
## Todo lo que se pueda guardar
# Desarrollar conjunto de pruebas propio
