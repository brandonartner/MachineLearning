#Convolutional Neural Network

#Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

#Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started

#Installing Keras
# pip install --upgrade keras

#Part 1 - Building the CNN

#Importing the keras libraries
from keras.models import Sequential # Used to initialize the NN (This package is used when the NN is a sequence of layers rather than a graph.)
from keras.layers import Conv2D # Used for the convolution step of the CNN, 2D since image, 3D would be for a video?
from keras.layers import MaxPooling2D # Used for the pooling step of the CNN
from keras.layers import Flatten # Used to convert the pooled images into feature vectors
from keras.layers import Dense # Used to add fully connected layers

#Initialize the CNN
classifier = Sequential()

#Step 1 - Convolution
classifier.add(Conv2D(32,(3,3), data_format='channels_last', input_shape=(64,64,3), activation='relu')) # Add convolution layer with 32 3x3 feature detectors/maps and 
																										  #		a rectifier activation function.

#Step 2 - Pooling
classifier.add(MaxPooling2D()) # Add a pooling layer using a 2x2 pool size (default) 

#Step 3 - Flattening
classifier.add(Flatten()) # Add a flattening layer that converts 2D pooled feature maps into a 1D feature vector

#Step 4 - Full Connection
classifier.add(Dense(units=128, activation='relu')) # Add a hidden layer with 128 nodes (128 was some optimized number the course creator found)
classifier.add(Dense(units=1, activation='sigmoid')) # Add the output layer with a sigmoid activation function (?) and only 1 unit (There are 2 outcomes shouldn't this be 2?)


#Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Compile CNN with ADAM optimizer, 
																						# binary cross entropy loss calculation (if more than 2 outcomes just crossentropy),
																						# and the metric being accuracy

#Part 2 - Fitting the CNN to the images
# Need to use some image preprocessing (data augmentation) to avoid over-fitting (The CNN fails to generalize from the training set), since the dataset is small
from keras.preprocessing.image import ImageDataGenerator

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
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Setup the test set
# Keras automatically detects the two classes (cat/dog) based on the folder structure in the given directory
test_generator = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

if input("Fit CNN to Data? (Y/N): ") == "Y"
	classifier.fit_generator(
		        train_generator,
		        steps_per_epoch=8000,
		        epochs=25,
		        validation_data=test_generator,
		        validation_steps=2000)