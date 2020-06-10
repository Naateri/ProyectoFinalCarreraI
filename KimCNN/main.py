from KimCNN import KimCNN
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

def transform(sentence, pad_len):
	vector = list()
	sentence = sentence.split()
	for word in sentence:
		if word.lower() not in words:
			vector.append(2)
		else:
			vector.append(words[word.lower()])
	# Saving as numpy array
	vector = np.array(vector) 
	# Saving as a matrix
	vector = np.array([vector])
	# Padding
	padded_vector = sequence.pad_sequences(vector, maxlen=pad_len)
	return padded_vector


train = False # True if training neural network, False if testing

maxlen = 400
max_features = 20000
network = KimCNN(400, max_features)

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

if train:
	print('Training model')
	print('Loading data...')
	data = imdb.load_data(num_words=max_features)
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

	print("x shape: ", x_train[0].shape)
		
	example = transform("That guy sucks really bad buddy", maxlen)
	print("example shape", example.shape)
	#print("example in array shape", np.array([example]).shape)

	network.train(x_train, x_test, y_train, y_test, batch_size=50, epochs=45,
			save_model=True, save_file='models/kimcnn_model')

	model = network.model

	print("Value to predict:", "That guy sucks really bad buddy")
	predictions = model.predict(np.array(example))
	print(predictions.shape)
	print(predictions)

	print("Value to predict:", "He is an amazing driver loved it")
	example = transform("He is an amazing driver loved it", maxlen)
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
		example = transform(predict, maxlen)
		predictions = model.predict(np.array(example))
		print(predictions)

		if (predictions[0][0] <= 0.5):
			print("Negative message")
		else:
			print("Positive message")
