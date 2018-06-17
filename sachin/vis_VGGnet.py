from keras.layers import Convolution2D, MaxPooling2D, Activation
from keras.models import Sequential
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2
import matplotlib.pyplot as plt 

# cat = cv2.imread('kit.png')
# shp = cat.shape
# print cat.shape[0]

#data_preprocess for keras
#keras always needs a batch input
#hence create a fake batch of sixe 1
# cat_batch = np.expand_dims(cat,axis=0)
# cat_batch = preprocess_input(cat_batch)

img = image.load_img('kit.png', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

#lets load the vgg19 model and play with it 

base_model = VGG16(weights='imagenet')
#print the summary of the model
#base_model.summary()
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

block4_pool_features = model.predict(x)
print np.shape(block4_pool_features)

model_output = np.squeeze(block4_pool_features, axis=0)
print np.shape(model_output)
# cv2.imshow('model output',model_output[:,:,10])
cv2.imshow('model output',model_output[:,:,20])
# cv2.imshow('model output',model_output[:,:,30])
# cv2.imshow('model output',model_output[:,:,40])
# cv2.imshow('model output',model_output[:,:,50])
cv2.waitKey()