import tensorflow as tf 
import numpy as np 
import pandas as pd
from keras.layers import LSTM

from keras.layers.core import Dense, Activation, Dropout

filepath = 'dataset.csv' 

def data_cleaning(dataframe):
	# for i in range(len(dataframe)):

	# 	row = dataframe.loc[i:i]
	print "data cleaning initiated"
	
	print "checking values of x and y coordinates"
	dataframe = dataframe[dataframe.x < 999]
	dataframe = dataframe[dataframe.y < 999]
	print "checked"

	dataframe = dataframe[dataframe.obj0 + dataframe.obj1 + dataframe.obj2 + dataframe.obj3 + dataframe.obj4 < 11]
	dataframe = dataframe[dataframe.obj0 + dataframe.obj1 + dataframe.obj2 + dataframe.obj3 + dataframe.obj4 > -1]
	
	print "checking ids of nodeMCU's"
	dataframe = dataframe[dataframe.obj0 < 6]
	dataframe = dataframe[dataframe.obj1 < 6]
	dataframe = dataframe[dataframe.obj2 < 6]
	dataframe = dataframe[dataframe.obj3 < 6]
	dataframe = dataframe[dataframe.obj4 < 6]

	dataframe = dataframe[dataframe.obj0 > -1]
	dataframe = dataframe[dataframe.obj1 > -1]
	dataframe = dataframe[dataframe.obj2 > -1]
	dataframe = dataframe[dataframe.obj3 > -1]
	dataframe = dataframe[dataframe.obj4 > -1]
	print 'checked'

	print "checking errors in received datapacket"
	dataframe = dataframe[dataframe.obj0 != dataframe.obj1]
	dataframe = dataframe[dataframe.obj0 != dataframe.obj2]
	dataframe = dataframe[dataframe.obj0 != dataframe.obj3]
	dataframe = dataframe[dataframe.obj0 != dataframe.obj4]
	
	dataframe = dataframe[dataframe.obj1 != dataframe.obj2]
	dataframe = dataframe[dataframe.obj1 != dataframe.obj3]
	dataframe = dataframe[dataframe.obj1 != dataframe.obj4]
	
	dataframe = dataframe[dataframe.obj2 != dataframe.obj3]
	dataframe = dataframe[dataframe.obj2 != dataframe.obj4]

	dataframe = dataframe[dataframe.obj3 != dataframe.obj4]
	print 'checked'

	RSSI_columns = ['obj0','rssi0','obj1','rssi1','obj2','rssi2','obj3','rssi3','obj4','rssi4']

	print "dataset filtered"
	return dataframe

data = pd.read_csv(filepath)

data = data_cleaning(data)

print(data.head())