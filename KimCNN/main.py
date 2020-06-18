from KimCNN import KimCNN
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import os

def transform(sentence, pad_len, words, UNK=2): # Pass sentence to integers
	vector = list()
	sentence = sentence.split()
	for word in sentence:
		if word.lower() not in words:
			vector.append(UNK)
		else:
			vector.append(words[word.lower()])
	# Saving as numpy array
	vector = np.array(vector) 
	# Saving as a matrix
	vector = np.array([vector])
	# Padding
	padded_vector = sequence.pad_sequences(vector, maxlen=pad_len)
	return padded_vector

def get_wordvec_imdb():
	vocab_file = 'datasets/aclimdb/imdb.vocab'

	vf = open(vocab_file, 'r') # vf = vocab_file
	# Create dictionary with words and its index
	words = dict()
	index = 0

	for word in vf:
		words[word.lower()] = index
		index += 1

	vf.close()

	return words, index

def load_imdb():
	train_pos_files = 'datasets/aclimdb/train/pos'
	train_neg_files = 'datasets/aclimdb/train/neg'
	test_pos_files = 'datasets/aclimdb/test/pos'
	test_neg_files = 'datasets/aclimdb/test/neg'
	
	train_x = list()
	train_y = list()
	test_x = list()
	test_y = list()

	words, index = get_wordvec_imdb()
	
	# train positives
	print("IMDB train positives")
	for filename in os.listdir(train_pos_files):
		#print(os.path.join(train_pos_files, filename))
		# open in read_only mode
		f = open(os.path.join(train_pos_files, filename), 'r')
		#print(f.read())
		array = transform(f.read(), 400, words, index)
		train_x.append(array)
		train_y.append(1.0)
		f.close()

	# train negatives
	print("IMDB train negatives")
	for filename in os.listdir(train_neg_files):
		#print(os.path.join(train_pos_files, filename))
		# open in read_only mode
		f = open(os.path.join(train_neg_files, filename), 'r')
		#print(f.read())
		array = transform(f.read(), 400, words, index)
		train_x.append(array)
		train_y.append(0.0)
		f.close()

	# test positives
	print("IMDB test positives")
	for filename in os.listdir(test_pos_files):
		#print(os.path.join(train_pos_files, filename))
		# open in read_only mode
		f = open(os.path.join(test_pos_files, filename), 'r')
		#print(f.read())
		array = transform(f.read(), 400, words, index)
		test_x.append(array)
		test_y.append(1.0)
		f.close()
	
	# test negatives
	print("IMDB test negatives")
	for filename in os.listdir(test_neg_files):
		#print(os.path.join(train_pos_files, filename))
		# open in read_only mode
		f = open(os.path.join(test_neg_files, filename), 'r')
		#print(f.read())
		array = transform(f.read(), 400, words, index)
		test_x.append(array)
		test_y.append(0.0)
		f.close()
	
	return (train_x, train_y), (test_x, test_y)
	

train = True # True if training neural network, False if testing

maxlen = 400
max_features = 20000
network = KimCNN(400, max_features)

#words = imdb.get_word_index()
#words = {k:(v+3) for k,v in words.items()}
#words["<PAD>"] = 0
#words["<START>"] = 1
#words["<UNK>"] = 2
#words["<UNUSED>"] = 3

words, _ = get_wordvec_imdb()

id_to_word = {value:key for key,value in words.items()}

get_sentence = lambda vector : ' '.join(id_to_word[num] for num in vector)

#print(get_sentence(x_train[0]), y_train[0])
#print(get_sentence(x_train[1]), y_train[1])

#load_imdb()
#exit()

if train:
	print('Training model')
	print('Loading data...')
	#data = imdb.load_data(num_words=max_features)
	data = load_imdb()
	(x_train, y_train), (x_test, y_test) = data

	# x_train, x_test = list of indexes
	# y_train, y_test = list of 0 (neg) or 1 (pos)

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
	example = transform("That guy sucks really bad buddy", maxlen, words)
	print("example shape", example.shape)
	#print("example in array shape", np.array([example]).shape)

	network.train(x_train, x_test, y_train, y_test, batch_size=50, epochs=1,
			save_model=True, save_file='models/kimcnn_model_imdb')

	model = network.model

	print("Value to predict:", "That guy sucks really bad buddy")
	predictions = model.predict(np.array(example))
	print(predictions.shape)
	print(predictions)

	print("Value to predict:", "He is an amazing driver loved it")
	example = transform("He is an amazing driver loved it", maxlen, words)
	predictions = model.predict(np.array(example))
	print(predictions)

else:
	model = network.load('models/kimcnn_model')
	print("Correctly loaded model")
	while True:
		predict = input('Type in text to analize ')
		if predict == '0':
			break
		
		print("Value to predict", predict)
		example = transform(predict, maxlen, words)
		predictions = model.predict(np.array(example))
		print(predictions)

		if (predictions[0][0] <= 0.5):
			print("Negative message")
		else:
			print("Positive message")
