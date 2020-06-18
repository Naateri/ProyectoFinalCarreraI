# Code based on:
# https://kaggle.com/hamishdickson/cnn-for-sentence-classification-by-yoon-kim
from __future__ import print_function
import numpy as np
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv2D, MaxPool2D, Reshape, Concatenate, Flatten
from tensorflow.keras.optimizers import RMSprop

class KimCNN:

	model = None

	def __init__(self, maxlen, max_features):

		num_filters = 100

		inputs = keras.Input(shape=(maxlen,), name="InputLayer") # Input
		embedding_layer = Embedding(max_features, 300, input_length=maxlen, 
			trainable=True, name="EmbeddingLayer")(inputs) # Embedding layer

		reshape = Reshape((maxlen, 300, 1)) (embedding_layer)

		conv_3 = Conv2D(num_filters, kernel_size=(3, 300), activation='relu',
			kernel_initializer='normal', 
			kernel_regularizer=keras.regularizers.l2(3)) (reshape)
		
		conv_4 = Conv2D(num_filters, kernel_size=(4,300), activation='relu',
			kernel_initializer='normal', 
			kernel_regularizer=keras.regularizers.l2(3)) (reshape)
		
		conv_5 = Conv2D(num_filters, kernel_size=(5,300), activation='relu',
			kernel_initializer='normal', 
			kernel_regularizer=keras.regularizers.l2(3)) (reshape)

		maxpool_3 = MaxPool2D(pool_size=(maxlen - 3 + 1, 1), strides=(1,1),
			padding='valid') (conv_3)

		maxpool_4 = MaxPool2D(pool_size=(maxlen - 4 + 1, 1), strides=(1,1),
			padding='valid') (conv_4)
		
		maxpool_5 = MaxPool2D(pool_size=(maxlen - 5 + 1, 1), strides=(1,1),
			padding='valid') (conv_5)

		concatenate_layer = Concatenate(axis=1)([maxpool_3, maxpool_4, maxpool_5])

		flatten_layer = Flatten()(concatenate_layer)

		dropout_layer = Dropout(0.5)(flatten_layer)

		outputs = Dense(1, activation='sigmoid') (dropout_layer)

		self.model = keras.Model(inputs=inputs, outputs=outputs)

		#self.model.summary()

		self.model.compile(optimizer='adam', loss='binary_crossentropy',
					metrics=['accuracy'])

	def train(self, x_train, x_test, y_train, y_test, /, batch_size=50, epochs=32, 
		save_model = False, save_file = 'models/kimcnn_model'):
		print('Train...')
		start = time.perf_counter()

		self.model.fit(x_train, y_train,
				batch_size=batch_size,
				epochs=epochs,
				validation_data=[x_test, y_test],
				verbose=1,
				validation_split=0.2)
				
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
