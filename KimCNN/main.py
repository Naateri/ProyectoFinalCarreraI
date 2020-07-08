## TODO:
## -word2vec
## -KFold for MR dataset

from KimCNN import KimCNN
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import os


# Pass sentence to integers
def transform(sentence, pad_len, words, /, ret_array=True, UNK=2,
		max_features=40000):
	vector = list()
	sentence = sentence.split()
	for word in sentence:
		if word.lower() not in words:
			vector.append(UNK)
		else:
			if words[word.lower()] >= max_features:
				vector.append(UNK)
			else:
				vector.append(words[word.lower()])
	# Saving as numpy array
	vector = np.array(vector) 
	# Saving as a matrix
	vector = np.array([vector])
	# Padding
	padded_vector = sequence.pad_sequences(vector, maxlen=pad_len)
	if ret_array:
		return padded_vector
	else:
		return padded_vector[0]

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

def load_MR(words, index):
	pos_files = 'datasets/review_polarity/txt_sentoken/pos'
	neg_files = 'datasets/review_polarity/txt_sentoken/neg'
	
	train_x = list()
	train_y = list()
	#words, index = get_wordvec_imdb() # For now

	# train positives
	print("MR positives")
	for filename in os.listdir(pos_files):
		#print(os.path.join(train_pos_files, filename))
		# open in read_only mode
		f = open(os.path.join(pos_files, filename), 'r')
		#print(f.read())
		array = transform(f.read(), 400, words, UNK=index, ret_array=False)
		train_x.append(array)
		train_y.append(1.0)
		f.close()

	print("MR negatives")
	for filename in os.listdir(neg_files):
		#print(os.path.join(train_pos_files, filename))
		# open in read_only mode
		f = open(os.path.join(neg_files, filename), 'r')
		#print(f.read())
		array = transform(f.read(), 400, words, UNK=index, ret_array=False)
		train_x.append(array)
		train_y.append(1.0)
		f.close()

	return train_x, train_y


def load_imdb(words, index):
	train_pos_files = 'datasets/aclimdb/train/pos'
	train_neg_files = 'datasets/aclimdb/train/neg'
	test_pos_files = 'datasets/aclimdb/test/pos'
	test_neg_files = 'datasets/aclimdb/test/neg'
	
	train_x = list()
	train_y = list()
	test_x = list()
	test_y = list()

	#words, index = get_wordvec_imdb()
	
	# train positives
	print("IMDB train positives")
	for filename in os.listdir(train_pos_files):
		#print(os.path.join(train_pos_files, filename))
		# open in read_only mode
		f = open(os.path.join(train_pos_files, filename), 'r')
		#print(f.read())
		array = transform(f.read(), 400, words, UNK=index, ret_array=False)
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
		array = transform(f.read(), 400, words, UNK=index, ret_array=False)
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
		array = transform(f.read(), 400, words, UNK=index, ret_array=False)
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
		array = transform(f.read(), 400, words, UNK=index, ret_array=False)
		test_x.append(array)
		test_y.append(0.0)
		f.close()
	
	return (train_x, train_y), (test_x, test_y)

def load_sst2(words, index):
	textfile = 'datasets/SST2/unsup.csv'
	trainfile = 'datasets/SST2/train.csv'
	testfile = 'datasets/SST2/test.csv'

	train_x = list()
	train_y = list()
	test_x = list()
	test_y = list()

	#words, index = get_wordvec_imdb()
	'''
	f = open(textfile, 'r')
	
	while True:
		line = f.readline()

		if not line:
			break
			
		cur_data = line.split(',')

		if cur_data[0] == 'label':
			continue

		text = cur_data[1].strip('\"\n')
		#print(text)
		array = transform(text, 400, words, UNK=index, ret_array=False)
		train_x.append(array)
		
		train_y.append(float(cur_data[0]))
	'''

	## Train

	f = open(trainfile, 'r')
	
	while True:
		line = f.readline()

		if not line:
			break
			
		cur_data = line.split(',')

		if cur_data[0] == 'label':
			continue

		text = cur_data[1].strip('\"\n')
		#print(text)
		array = transform(text, 400, words, UNK=index, ret_array=False)
		train_x.append(array)
		
		train_y.append(float(cur_data[0]))
	
	f.close()

	## Test
	f = open(testfile, 'r')
	
	while True:
		line = f.readline()

		if not line:
			break
			
		cur_data = line.split(',')

		if cur_data[0] == 'label':
			continue

		text = cur_data[1].strip('\"\n')
		#print(text)
		array = transform(text, 400, words, UNK=index, ret_array=False)
		test_x.append(array)
		
		test_y.append(float(cur_data[0]))

	return (train_x, train_y), (test_x, test_y)
	

train = True # True if training neural network, False if testing

maxlen = 400
max_features = 40000

imdb_words = imdb.get_word_index()
imdb_words = {k:(v+3) for k,v in imdb_words.items()}
imdb_words["<PAD>"] = 0
imdb_words["<START>"] = 1
imdb_words["<UNK>"] = 2
imdb_words["<UNUSED>"] = 3

words, imdb_index = get_wordvec_imdb()

id_to_word = {value:key for key,value in words.items()}

get_sentence = lambda vector : ' '.join(id_to_word[num] for num in vector)

#print(get_sentence(x_train[0]), y_train[0])
#print(get_sentence(x_train[1]), y_train[1])

#load_imdb()
#exit()

DATASET = 3

# 0 -> keras imdb
# 1 -> IMDB
# 2 -> MR 
# 3 -> SST2

if train:
	print('Training model')
	print('Loading data...')
	if DATASET == 0:
		data = imdb.load_data(num_words=max_features)
		(x_train, y_train), (x_test, y_test) = data
	elif DATASET == 1:
		#data = load_imdb(words, imdb_index)
		data = load_imdb(imdb_words, 2)
		(x_train, y_train), (x_test, y_test) = data
	elif DATASET == 2:
		print("len(i_words)", len(imdb_words))
		data = load_MR(imdb_words, 2)
		X, y = data
		x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	elif DATASET == 3:
		data = load_sst2(imdb_words, 2)
		(x_train, y_train), (x_test, y_test) = data

	# x_train, x_test = list of indexes (words)
	# y_train, y_test = list of 0 (neg) or 1 (pos)

	print(len(x_train), 'train sequences')
	print(len(x_test), 'test sequences')

	print('Pad sequences (samples x time)')
	if DATASET == 0:
		x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
		x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
	
	x_train = np.array(x_train)
	x_test = np.array(x_test)
	print('x_train shape:', x_train.shape)
	print('x_test shape:', x_test.shape)
	y_train = np.array(y_train)
	y_test = np.array(y_test)

	print("x shape: ", x_train[0].shape)
	print("x_train[0]", x_train[0])
	print("x_train[1]", x_train[1])
	example = transform("That guy sucks really bad buddy", maxlen, imdb_words)
	print("example shape", example.shape)
	#print("example in array shape", np.array([example]).shape)

	batch_size = 50
	if DATASET == 0 or DATASET == 1:
		network = KimCNN(400, max_features)
		#epochs = 45
		epochs = 1
	elif DATASET == 2:
		network = KimCNN(400, max_features)
		batch_size = 20
		#epochs = 60
		epochs = 1
	elif DATASET == 3:
		network = KimCNN(400, max_features)
		epochs = 10

	print("Ready to train")
	#exit()
	if DATASET == 0:
		network.train(x_train, x_test, y_train, y_test, batch_size=batch_size, epochs=epochs,
			save_model=True, save_file='models/kimcnn_model')
	elif DATASET == 1:
		network.train(x_train, x_test, y_train, y_test, batch_size=batch_size, epochs=epochs,
			save_model=True, save_file='models/kimcnn_model_imdb')
	elif DATASET == 2:
		network.train(x_train, x_test, y_train, y_test, batch_size=batch_size, epochs=epochs,
			save_model=True, save_file='models/kimcnn_model_MR')
	elif DATASET == 3:
		network.train(x_train, x_test, y_train, y_test, batch_size=batch_size, epochs=epochs,
			save_model=True, save_file='models/kimcnn_model_SST2')

	model = network.model

	print("Value to predict:", "That guy sucks really bad buddy")
	predictions = model.predict(np.array(example))
	print(predictions.shape)
	print(predictions)

	print("Value to predict:", "He is an amazing driver loved it")
	example = transform("He is an amazing driver loved it", maxlen, imdb_words)
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
