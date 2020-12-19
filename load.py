import numpy as np
import keras.models
from keras.models import model_from_json
import tensorflow as tf


def init(): 
	json_file = open('model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("model.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	#loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	return loaded_model