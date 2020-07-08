from HameedBiLSTM import HameedBiLSTM
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

### Global Variables ###

DATASET = 3
# 0 -> keras IMDB
# 1 -> MR
# 2 -> SST2
# 3 -> IMDB

words = imdb.get_word_index()
words = {k:(v+3) for k,v in words.items()}
words["<PAD>"] = 0
words["<START>"] = 1
words["<UNK>"] = 2
words["<UNUSED>"] = 3

id_to_word = {value:key for key,value in words.items()}

get_sentence = lambda vector : ' '.join(id_to_word[num] for num in vector)

maxlen = 400
max_features = 20000

def transform(sentence, pad_len, words=words):
	vector = list()
	sentence = sentence.split()
	for word in sentence:
		try:
			vector.append(words[word.lower()])
		except:
			vector.append(2)
		'''
		if word.lower() in words:
			if words[word.lower()] < max_features:
				vector.append(words[word.lower()])
			else:
				vector.append(2)
		else:
			vector.append(2)'''
	# Saving as numpy array
	vector = np.array(vector) 
	# Saving as a matrix
	vector = np.array([vector])
	# Padding
	padded_vector = sequence.pad_sequences(vector, maxlen=pad_len)
	return padded_vector

### DATASETS ###

def load_MR(pad_len=40):
	pos_files = 'datasets/MR/rt-polarity.pos'
	neg_files = 'datasets/MR/rt-polarity.neg'
	
	train_x = list()
	train_y = list()
	texts = list()

	# train positives
	print("MR positives")

	f = open(pos_files, 'rb')
	i = 0

	for line in f:
		try:
			cur_line = line.decode('ascii').replace('\n', '')
		except:
			#print('error at', i)
			#cur_line = line.decode('ascii', 'replace').replace('\n', '')
			#print(cur_line)
			#i += 1
			continue
		text_converted = transform(cur_line, pad_len)
		#train_x.append(text_converted[0])
		texts.append(cur_line)
		train_y.append(1.0)
		i += 1
	
	f.close()
	
	print("MR negatives")
	
	f = open(neg_files, 'rb')
	i = 0

	for line in f:
		try:
			cur_line = line.decode('ascii').replace('\n', '')
		except:
			#print('error at', i)
			#cur_line = line.decode('ascii', 'replace').replace('\n', '')
			#print(cur_line)
			#i += 1
			continue
		text_converted = transform(cur_line, pad_len)
		#train_x.append(text_converted[0])
		texts.append(cur_line)
		train_y.append(0.0)
		i += 1

	# Creating dictionary
	#tokenizer = Tokenizer(num_words=max_features)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)

	word_index = tokenizer.word_index
	train_x = sequence.pad_sequences(sequences, maxlen=maxlen)

	return train_x, train_y, word_index

def load_SST2(pad_len=45):
	alltexts = 'datasets/SST2/unsup.csv'
	trainfile = 'datasets/SST2/train.csv'
	testfile = 'datasets/SST2/test.csv'
	valfile = 'datasets/SST2/val.csv'

	temp_train_x = list()
	train_y = list()
	temp_test_x = list()
	test_y = list()

	texts = list()

	## Build tokenizer

	print('Creating SST2 word index')

	f = open(alltexts, 'r')

	for line in f:
		cur_data = line.split(',')

		# Ignore first line
		if cur_data[0] == 'label':
			continue

		text = cur_data[1].strip('\"\n')
		texts.append(text)

	f.close()

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(texts)

	word_index = tokenizer.word_index
	
	## train and validation = train data

	print('SST2 train data')

	f = open(trainfile, 'r')

	for line in f:
		cur_data = line.split(',')

		if cur_data[0] == 'label':
			continue

		text = cur_data[1].strip('\"\n')
		#print(text)
		temp_train_x.append(text)
		
		train_y.append(float(cur_data[0]))
	
	f.close()

	## Test
	print('SST2 val data')
	f = open(valfile, 'r')

	for line in f:
		cur_data = line.split(',')

		if cur_data[0] == 'label':
			continue

		text = cur_data[1].strip('\"\n')
		#print(text)
		temp_test_x.append(text)
		
		test_y.append(float(cur_data[0]))

	f.close()
	
	sequences = tokenizer.texts_to_sequences(temp_train_x)
	train_x = sequence.pad_sequences(sequences, maxlen=maxlen)

	test_seq = tokenizer.texts_to_sequences(temp_test_x)
	test_x = sequence.pad_sequences(test_seq, maxlen=maxlen)

	return (train_x, train_y), (test_x, test_y), word_index

def load_SST2test(word_index):
	testfile = 'datasets/SST2/test.csv'

	f = open(testfile, 'r')
	x = list()
	y = list()

	for line in f:
		cur_data = line.split(',')

		if cur_data[0] == 'label':
			continue

		text = cur_data[1].strip('\"\n')

		array = transform(text, 45, word_index)
		#print(text)
		x.append(array[0])
		
		y.append(float(cur_data[0]))

	f.close()

	return x, y

def load_imdb():
	train_pos_files = 'datasets/aclimdb/train/pos'
	train_neg_files = 'datasets/aclimdb/train/neg'
	test_pos_files = 'datasets/aclimdb/test/pos'
	test_neg_files = 'datasets/aclimdb/test/neg'
	all_files = 'datasets/aclimdb/train/unsup'
	
	temp_train_x = list()
	train_y = list()
	temp_test_x = list()
	test_y = list()

	texts = list()

	## Build tokenizer

	print('Creating IMDB word index')

	for filename in os.listdir(all_files):
		#print(os.path.join(train_pos_files, filename))
		# open in read_only mode
		f = open(os.path.join(all_files, filename), 'r')
		
		text = f.read()
		text = text.replace('<br /><br />', '')
		texts.append(text)
		f.close()

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(texts)

	word_index = tokenizer.word_index
	
	# train positives
	print("IMDB train positives")
	for filename in os.listdir(train_pos_files):
		#print(os.path.join(train_pos_files, filename))
		# open in read_only mode
		f = open(os.path.join(train_pos_files, filename), 'r')
		#f = open(train_pos_files+"/"+filename, 'r')
		#print(f.read())
		text = f.read()
		text = text.replace('<br /><br />', '')
		temp_train_x.append(text)
		train_y.append(1.0)
		f.close()

	# train negatives
	print("IMDB train negatives")
	for filename in os.listdir(train_neg_files):
		#print(os.path.join(train_pos_files, filename))
		# open in read_only mode
		f = open(os.path.join(train_neg_files, filename), 'r')
		#print(f.read())
		text = f.read()
		text = text.replace('<br /><br />', '')
		temp_train_x.append(text)
		train_y.append(0.0)
		f.close()

	# test positives
	print("IMDB test positives")
	for filename in os.listdir(test_pos_files):
		#print(os.path.join(train_pos_files, filename))
		# open in read_only mode
		f = open(os.path.join(test_pos_files, filename), 'r')
		#print(f.read())
		text = f.read()
		text = text.replace('<br /><br />', '')
		temp_test_x.append(text)
		test_y.append(1.0)
		f.close()
	
	# test negatives
	print("IMDB test negatives")
	for filename in os.listdir(test_neg_files):
		#print(os.path.join(train_pos_files, filename))
		# open in read_only mode
		f = open(os.path.join(test_neg_files, filename), 'r')
		#print(f.read())
		text = f.read()
		text = text.replace('<br /><br />', '')
		temp_test_x.append(text)
		test_y.append(0.0)
		f.close()

	sequences = tokenizer.texts_to_sequences(temp_train_x)
	train_x = sequence.pad_sequences(sequences, maxlen=maxlen)

	test_seq = tokenizer.texts_to_sequences(temp_test_x)
	test_x = sequence.pad_sequences(test_seq, maxlen=maxlen)

	return (train_x, train_y), (test_x, test_y), word_index

train = True # True if training neural network, False if testing

if DATASET == 0:
	network = HameedBiLSTM(400, max_features, use_glove=True, word_index=words)
	maxlen = 400
elif DATASET == 1: # MR
	#max_features = len(words)
	#network = HameedBiLSTM(40, max_features, use_glove=False, word_index=words)
	maxlen = 40
elif DATASET == 2:
	maxlen = 45
elif DATASET == 3:
	maxlen = 400

#print(get_sentence(x_train[0]), y_train[0])
#print(get_sentence(x_train[1]), y_train[1])

if train:
	print('Training model')
	print('Loading data...')
	if DATASET == 0: # keras IMDB
		data = imdb.load_data(num_words=max_features)
		(x_train, y_train), (x_test, y_test) = data
	
	elif DATASET == 1: # MR
		X, y, mr_wordindex = load_MR()
		print('Total data: ', len(X))
		x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

		max_features = len(mr_wordindex)+1

		network = HameedBiLSTM(40, max_features, use_glove=True, word_index=mr_wordindex)
	
	elif DATASET == 2: # SST2
		(x_train, y_train), (x_test, y_test), sst2_wordindex = load_SST2()
		print('Total data: ', len(x_train) + len(x_test))

		max_features = len(sst2_wordindex) + 1

		network = HameedBiLSTM(45, max_features, use_glove=True, word_index=sst2_wordindex)
	
	elif DATASET == 3: # IMDB
		(X, y), (x_val, y_val), imdb_wordindex = load_imdb()
		print('Total data: ', len(X) + len(x_val))
		# val will be used to test the models accuracy
		# based on the split made by the dataset

		x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

		max_features = len(imdb_wordindex) + 1

		network = HameedBiLSTM(400, max_features, use_glove=True, word_index=imdb_wordindex)
	
	print(len(x_train), 'train sequences')
	print(len(x_test), 'test sequences')
	

	print('Pad sequences (samples x time)')
	x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
	x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
	print('x_train shape:', x_train.shape)
	print('x_test shape:', x_test.shape)
	y_train = np.array(y_train)
	y_test = np.array(y_test)

	print("x shape: ", x_train[0].shape)
	
	if DATASET == 0:
		example = transform("That guy sucks really bad buddy", maxlen)
	elif DATASET == 1:
		example = transform("That guy sucks really bad buddy", maxlen, mr_wordindex)
	elif DATASET == 2:
		example = transform("That guy sucks really bad buddy", maxlen, sst2_wordindex)
	elif DATASET == 3:
		example = transform("That guy sucks really bad buddy", maxlen, imdb_wordindex)

	print("example shape", example.shape)
	#print("example in array shape", np.array([example]).shape)
	if DATASET == 0:
		network.train(x_train, x_test, y_train, y_test, batch_size=64, epochs=45,
			save_model=False, save_file='models/bilstm_model_glove')
	elif DATASET == 1: # MR
		epochs = 70
		network.train(x_train, x_test, y_train, y_test, batch_size=64, epochs=epochs,
			save_model=True, save_file='models/bilstm_model_MR_glove_test10T')
	elif DATASET == 2: # SST2
		epochs = 70
		network.train(x_train, x_test, y_train, y_test, batch_size=64, epochs=epochs,
			save_model=True, save_file='models/bilstm_model_SST2_glove')
	elif DATASET == 3: # IMDB
		epochs = 45
		network.train(x_train, x_test, y_train, y_test, batch_size=64, epochs=3,
			save_model=False, save_file='models/bilstm_model_IMDB_glove')

	model = network.model

	print("Value to predict:", "That guy sucks really bad buddy")
	predictions = model.predict(np.array(example))
	print(predictions.shape)
	print(predictions)
	#print(np.argmax(predictions, axis=1))

	print("Value to predict:", "He is an amazing driver loved it")
	if DATASET == 0:
		example = transform("He is an amazing driver loved it", maxlen)
	elif DATASET == 1:
		example = transform("He is an amazing driver loved it", maxlen, mr_wordindex)
	elif DATASET == 2:
		example = transform("He is an amazing driver loved it", maxlen, sst2_wordindex)
	elif DATASET == 3:
		example = transform("He is an amazing driver loved it", maxlen, imdb_wordindex)
	predictions = model.predict(np.array(example))
	print(predictions)
	#print(np.argmax(predictions, axis=1))

	if DATASET == 0:
		pass
	elif DATASET == 1:
		y_pred = model.predict(x_test)
		score, acc = model.evaluate(x_test, y_test, batch_size=64, verbose=0)

		temp = list()
		for value in y_pred:
			if value <= 0.5:
				temp.append(0.0)
			else:
				temp.append(1.0)
		
		y_pred = np.array(temp)

		conf_mat = confusion_matrix(y_test, y_pred)

		print('accuracy', acc)
		print('matrix', conf_mat)
	elif DATASET == 2:

		## val = test set

		x_val, y_val = load_SST2test(sst2_wordindex)
		x_val = np.array(x_val)
		y_val = np.array(y_val)
		y_pred = model.predict(x_val)
		#acc = tf.keras.metrics.binary_accuracy(y_val, y_pred, threshold=0.5)

		score, acc = model.evaluate(x_val, y_val, batch_size=64, verbose=0)

		temp = list()
		for value in y_pred:
			if value <= 0.5:
				temp.append(0.0)
			else:
				temp.append(1.0)
		
		y_pred = np.array(temp)

		conf_mat = confusion_matrix(y_val, y_pred)

		print('accuracy', acc)
		print('matrix', conf_mat)
	
	elif DATASET == 3: # IMDB
		x_val = np.array(x_val)
		y_val = np.array(y_val)
		y_pred = model.predict(x_val)
		#acc = tf.keras.metrics.binary_accuracy(y_val, y_pred, threshold=0.5)

		score, acc = model.evaluate(x_val, y_val, batch_size=64, verbose=0)

		temp = list()
		for value in y_pred:
			if value <= 0.5:
				temp.append(0.0)
			else:
				temp.append(1.0)
		
		y_pred = np.array(temp)

		conf_mat = confusion_matrix(y_val, y_pred)

		print('accuracy', acc)
		print('matrix', conf_mat)

else:
	model = network.load('models/bilstm_model')
	print("Correctly loaded model")
	while True:
		predict = input('Type in text to analize ')
		if predict == '0':
			break
		
		print("Value to predict", predict)
		example = transform(predict, maxlen)
		predictions = model.predict(np.array(example))
		print(predictions)

		if (predictions[0][0] <= 0.5):
			print("Negative message")
		else:
			print("Positive message")