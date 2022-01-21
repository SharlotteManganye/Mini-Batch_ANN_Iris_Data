
"""
-- *********************************************
-- Author       :	Sharlotte Manganye
-- Create date  :   21 January 2022
-- Description  :  ANN on Iris data
-- File Name    :  ANN.py
--*********************************************
"""

''' Load Libraries '''

from sklearn.datasets import load_iris
from tensorflow.keras.layers import  Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

'''  Loading dataset'''
df = load_iris()

'''  Parameters '''
SPLIT_SIZE = 0.2   #training, test and validation set split
bs = 16  #Batch size
lr = 0.01 #learning rate
units = 10 #number of units in the hidden layer
iterations = 100  #epochs
seed =42


'''Preprocessing data'''
'''prepare train and test dataset'''


def prepare_data():
	'''generate 2d classification dataset'''
	X, y = (df.data, df.target)
	''' Normalize the input data'''
	X = sc.fit_transform(X)
	''' one hot encode output variable'''
	y = to_categorical(y)
	''' split into train and test'''
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=SPLIT_SIZE,random_state=0)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=SPLIT_SIZE, random_state=0)
	return x_train, y_train, x_val, y_val


'''Fit Model '''


def fit_model(x_train, y_train, x_val, y_val):
	n_input = x_train.shape[1]
	hidden_units = n_input
	initializer = keras.initializers.HeNormal()  # He weights initialization
	n_output = y_train.shape[1]
	'''The model expects rows of data with n_input variables (the input_dim=n_input argument)
		 The first hidden layer has n_input+1 nodes and uses the relu activation function.
		The output layer has y_train.shape[1] node and uses the relu activation function.'''
	model = Sequential()
	model.add(Dense(hidden_units, input_dim=n_input, kernel_initializer=initializer,
					activation='relu'))  # input, with hidden unit=nodes
	model.add(Dense(n_output, activation='relu'))  # output
	# compile model
	opt = keras.optimizers.Adam(learning_rate=lr)
	model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
	# fit model
	history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=iterations, verbose=0, batch_size=bs)
	#     evaluate model

	_, train_acc = model.evaluate(x_train, y_train, verbose=0)
	_, test_acc = model.evaluate(x_val, y_val, verbose=0)
	print('Train Accuracy: %.3f, Test Accuracy: %.3f' % (train_acc, test_acc))
	# for all attributes stored in history
	#     print(history.history.keys())

	# print Collective mean

	#collective_mean = print("Collective Mean Fitness (classification error):", sum(history.history['loss']) / iterations)
	# print generalazation factor for overfitting, rho<1

	#rho = print("Generalization Factor (overfitting?):", sum(history.history['val_loss']) / sum(history.history['loss']))

	# plot training history
	pyplot.plot(history.history['accuracy'], label='train')
	pyplot.plot(history.history['val_accuracy'], label='test')
	pyplot.legend()
	pyplot.show()
	# plot training history loss ( error)
	pyplot.plot(history.history['loss'], label='train_loss')
	pyplot.plot(history.history['val_loss'], label='val_loss')
	pyplot.legend()
	pyplot.show()

'''Training Model'''
x_train, y_train, x_val, y_val = prepare_data()

fit_model(x_train, y_train, x_val, y_val)

pyplot.show()