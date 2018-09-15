# CNN Tuning and optimization attempt

#Importing the keras libraries
import keras
from keras.models import Sequential # Used to initialize the NN (This package is used when the NN is a sequence of layers rather than a graph.)
from keras.layers import Conv2D # Used for the convolution step of the CNN, 2D since image, 3D would be for a video?
from keras.layers import MaxPooling2D # Used for the pooling step of the CNN
from keras.layers import Flatten # Used to convert the pooled images into feature vectors
from keras.layers import Dense # Used to add fully connected layers
from keras.layers import Dropout
from keras import backend
from keras.callbacks import Callback

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterGrid

from keras.models import model_from_json

import os
import datetime
import logging		

class LossHistory(Callback):
	def __init__(self):
		super().__init__()
		self.epoch_id = 0

	def on_train_begin(self, logs):
		logging.info('Training began at {}...'.format(datetime.datetime.now()))

	def on_train_end(self, logs):
		self.epoch_id = 0

	def on_epoch_end(self, epoch, logs={}):
		self.epoch_id += 1
		logging.info("Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}"\
					.format(self.epoch_id, logs.get('acc'), logs.get('val_acc')))


# function to build the ANN
def build_classifier(optimizer,neurons_per_layer, hidden_layers):
	classifier = Sequential()
	classifier.add(Conv2D(32,(3,3), data_format='channels_last', input_shape=(*image_size,3), activation='relu')) # Add convolution layer with 32 3x3 feature detectors/maps and a rectifier activation function.
	classifier.add(MaxPooling2D()) # Add a pooling layer using a 2x2 pool size (default) 

	classifier.add(Conv2D(32,(3,3), data_format='channels_last', activation='relu')) # Add convolution layer with a rectifier activation function.
	classifier.add(MaxPooling2D()) # Add a pooling layer using a 2x2 pool size (default)

	classifier.add(Flatten()) # Add a flattening layer that converts 2D pooled feature maps into a 1D feature vector

	for _ in range(hidden_layers):
		classifier.add(Dense(units=neurons_per_layer, activation='relu')) # Add a hidden layer with 128 nodes (128 was some optimized number the course creator found)
		classifier.add(Dropout(rate=0.3))                                                  # Add dropout to layer with 10% of the layer's neurons 

	classifier.add(Dense(units=1, activation='sigmoid')) # Add the output layer with a sigmoid activation function (?) and only 1 unit (There are 2 outcomes shouldn't this be 2?)

	classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	return classifier


def test_params(batch_size=32, epochs=1, hidden_layers=1, optimizer='adam', neurons_per_layer='5'):

	classifier = build_classifier(optimizer, neurons_per_layer, hidden_layers)


	# Include some random image augmentations into the training set and scale the pixel values down between 0 and 1
	train_datagen = ImageDataGenerator(
	        rescale=1./255,
	        shear_range=0.2,
	        zoom_range=0.2,
	        horizontal_flip=True)

	# Scale the pixel values down between 0 and 1
	test_datagen = ImageDataGenerator(rescale=1./255)

	# Setup the training set 
	# Keras automatically detects the two classes (cat/dog) based on the folder structure in the given directory
	train_generator = train_datagen.flow_from_directory(
	        'dataset/training_set',
	        target_size=image_size,
	        batch_size=batch_size,
	        class_mode='binary')

	# Setup the test set
	# Keras automatically detects the two classes (cat/dog) based on the folder structure in the given directory
	test_generator = test_datagen.flow_from_directory(
	        'dataset/test_set',
	        target_size=image_size,
	        batch_size=batch_size,
	        class_mode='binary')

	history = LossHistory()

	classifier.fit_generator(
		        train_generator,
		        steps_per_epoch=(traing_set_size/batch_size),
		        epochs=epochs,
		        validation_data=test_generator,
		        validation_steps=(test_set_size/batch_size),
		        workers=12,
		        max_queue_size=100,
		        callbacks=[history])

	return classifier


def GridSearch(param_grid):
	logging.info("Grid Search ... {}\n".format(datetime.datetime.now()))

	param_grid = ParameterGrid(param_grid)

	for params in param_grid:
		logging.info("Model Parameters: {}".format(str(params)))

		classifier = test_params(params.get('batch_size'), 
									params.get('epochs'),
									params.get('hidden_layers'),
									params.get('optimizer'),
									params.get('neurons_per_layer'))
	return classifier


if __name__ == '__main__':

	image_size = (128,128)
	traing_set_size = 8000
	test_set_size = 2000
	logging_file = "log_file.log"

	logging.basicConfig(filename=logging_file, level=logging.INFO)

	params = {'batch_size':[16,32,64], 
				'epochs':[25,50], 
				'hidden_layers':[1,2],
				'optimizer':['adam','SGD'],
				'neurons_per_layer':[128]}

	best_params = {'batch_size':[64], 
					'epochs':[50], 
					'hidden_layers':[1],
					'optimizer':['adam'],
					'neurons_per_layer':[128]}
	classifier = GridSearch(best_params)


	if input("Save CNN ? (Y/N): ").lower() == "y":
		# Serialize model in JSON
		classifier_json = classifier.to_json()
		with open("classifier.json", "w") as json_file:
			json_file.write(classifier_json)

		# Serialize weights into HDF5
		classifier.save_weights("classifier.h5")
		logging.info("Saved Model!")