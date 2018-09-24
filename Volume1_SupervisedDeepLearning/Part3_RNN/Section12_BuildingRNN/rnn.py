# Recurrent Neural Network
# Implements a LSTM to predict future stock of Google.

### Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing the training set
#	Import data as data frame
#	Convert to numpy array since keras only takes numpy arrays
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values # Get the open stock price of google from the training set and turn it into a numpy array

# Feature Scaling (Could use either Standardisation or Normilisation) Use Normilisation for RNNs especially when there is a sigmoid function being used
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1), copy=True) # Note: these are the default values, so are unnessecary
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output (number of timesteps is important for overfitting and accuratacy of predictions)
#	i.e. it will use the past 60 days of stock prices to predict the next day (60 was experimentally found by instructor)
X_train = []
Y_train = []

timesteps = 60

# Make a 2D array containing each sequence of 60 days in the dataset and another array of the 61st day after each sequence in X_train
for i in range(timesteps, len(training_set_scaled)):
	X_train.append(training_set_scaled[i - timesteps : i, 0])
	Y_train.append(training_set_scaled[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshaping (add more dimensions to X_train/Y_train, in order to include more indicators i.e. trading amount)
#------------- IMPORTANT: This is a good spot to be improved by including more indicators. ------------------
number_of_indicators = 1
X_train = np.reshape(X_train, (*X_train.shape, number_of_indicators))


### Part 2 - Building the RNN

# Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN
regressor = Sequential() # Regression is used for predicting a continuous value and Classification (used in previous NNs) is for predicting category or class.
lstm_layer_neurons = 50

# Add the first LSTM layer and some Dropout regularization
	# LSTM Layer, 
	#	input: 	units - # of LSTM units/modules that will be used, 50 is used here to abtain high-dimensionality?
	#			return_sequences = True - Whether to return the last output in the output sequence, or the full sequence.
	#									, default is False and is used when there are no more LSTM layers after this one.
	#			input_shape - shape of data (3D), only needs the second two dimensions of the X_train 3-tuple (The first one is how many data points and not their shape)
regressor.add(LSTM(units=lstm_layer_neurons, return_sequences=True, input_shape=X_train.shape[1:]))

regressor.add(Dropout(rate=0.2)) # Dropout Regularization Layer, 


# Add the second LSTM layer and some Dropout regularization
	# LSTM Layer, 
	#	input: 	units - # of LSTM units/modules that will be used, 50 is used here to abtain high-dimensionality?
	#			return_sequences = True - Whether to return the last output in the output sequence, or the full sequence.
	#									, default is False and is used when there are no more LSTM layers after this one.
	#			input_shape - shape of data (3D), not needed since this is a hidden layer
regressor.add(LSTM(units=lstm_layer_neurons, return_sequences=True))
regressor.add(Dropout(rate=0.2)) # Dropout Regularization Layer, 


# Add the third LSTM layer and some Dropout regularization
regressor.add(LSTM(units=lstm_layer_neurons, return_sequences=True))
regressor.add(Dropout(rate=0.2)) # Dropout Regularization Layer, 

# Add the fourth LSTM layer and some Dropout regularization
	# LSTM Layer, 
	#	input: 	units - # of LSTM units/modules that will be used, 50 is used here to abtain high-dimensionality?
	#			return_sequences - No longer needed, left to be default, because this is the last layer so no more return sequences
regressor.add(LSTM(units=lstm_layer_neurons))
regressor.add(Dropout(rate=0.2)) # Dropout Regularization Layer, 

# Add the ouptpur layer
regressor.add(Dense(units=1))

# Compiling the RNN
	# rmsprop is recommended for RNNs and mean squared error is needed for continuous predictions
regressor.compile(optimizer='rmsprop', loss='mean_squared_error')


# Fitting the RNN to Training set
	# Fit Parameters: 
	#	training_set - 
	#	test_set - 
	#	epochs - 100 was found by the instructor experimenting
	#	batch_size - 32 for some reason
regressor.fit(X_train, Y_train, epochs=1, batch_size=32)


### Part 3 - Making the predictions and visualizing the results

# Getting the real stock price of 2017 from the test set
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
actual_stock_price = dataset_test.iloc[:, 1:2].values


# Gettin the predicted stock price of 2017
#-------------- IMPORTANT: Rewatch video Section 15, Lecture 81 to actually understand this part (I was starting to zone out) -------
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)

inputs = dataset_total[-(len(dataset_test)+timesteps):].values
inputs = inputs.reshape(-1,1) # Flipping? Maybe?
inputs = sc.transform(inputs) # Do not fit the test data, needs to be the same fitting as the training data, so just transform

X_test = []

for i in range(timesteps, len(inputs)):
	X_test.append(inputs[i - timesteps : i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (*X_test.shape, number_of_indicators))

predicted_stock_price = regressor.predict(X_test) # Make the prediction

predicted_stock_price = sc.inverse_transform(predicted_stock_price) # Invert the scaling done before, to get the actual values

# Visualize the results
plt.plot(actual_stock_price, color='red', label='Actual Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction Comparison')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.savefig('Google_Prediction.png')
