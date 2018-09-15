# Use a pre-existing CNN to predict whether or not an image is a dog or cat.
from keras.models import model_from_json, load_model
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os

image_height, image_width = 128,128
sub_directory = '/dataset/single_prediction/'

#Part 1 - Setup the CNN
# load json and create model
json_file = open('classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("classifier.h5")
print("Loaded model from disk")

#Part 2 - Preprocess the image
predict_datagen = ImageDataGenerator(rescale=1./255)

for filename in os.listdir(os.getcwd() + sub_directory):
	try:
		img = cv2.imread(os.getcwd()+sub_directory+filename)
		img = cv2.resize(img, (image_height, image_width))

		img = np.array(img).reshape((1,image_height, image_width,3))

		prediction = loaded_model.predict(img)

		if round(prediction[0][0]):
			result = 'dog'
		else:
			result = 'cat'

		print("{} contains a {}.".format(filename, result))

	except Exception as e:
		raise e
