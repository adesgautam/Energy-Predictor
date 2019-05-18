
# For the DL part
# from math import sqrt
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import keras
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.backend import clear_session

import requests 
import json
import time

# Django specific
from django.shortcuts import render
from django.conf import settings

# import urllib as urllib2
import io
import urllib, base64

# Get data
def get_data():
	url = "http://api.eia.gov/series/?api_key=31c285edf7b11b33e6703e95c142d66c&series_id=EBA.FLA-ALL.D.H"
	r = requests.get(url).text

	data = json.loads(r)
	data = data["series"][0]["data"]
	dataset = pd.DataFrame(data, columns=['DateTime', 'Energy (mWh)'])
	dataset['DateTime'] = pd.to_datetime(dataset['DateTime'], infer_datetime_format=True)
	dataset = dataset[::-1]
	dataset.set_index('DateTime', inplace=True)
	dataset = dataset.astype('float64')
	return dataset

def forecast(request):
	dataset = get_data()

	print("No. of samples",len(dataset))
	last_length = len(dataset)

	# while(True):
		# wait for x time
		# print("Sleeping for 1.5 hours...")
		# hours = 1.5
		# seconds = 60*60*hours
		# time.sleep(2)

		# # check again
		# dataset = get_data()

		# print("No. of samples",len(dataset))
		# new_length = len(dataset)

		# if new_length>=last_length:
	print("Last Length: ",last_length)
	# print("New Length: ", new_length)
	print("Training started...")

	# Hyperparameters
	timesteps = 1
	features = 1
	batch_size=1
	model_path = settings.MEDIA_ROOT + '/models/model.h5'
	filepath = model_path

	# removing NaNs
	for j in range(0,1):        
		dataset.iloc[:,j]=dataset.iloc[:,j].fillna(dataset.iloc[:,j].mean())

	values = dataset.values

	# normalize features
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)

	# frame the data
	reframed = series_to_supervised(scaled, timesteps, 1)
	print(reframed.iloc[-1,:])
	X, y = reframed.iloc[-1,:][0], reframed.iloc[-1,:][1]
	print(X, y)
	
	X = X.reshape((1, timesteps, features))
	y = y.reshape((1, ))			
	print(X.shape, y.shape) 
	# shape [samples, timesteps, features].

	# Load model (stateless)
	global model
	model = load_model(model_path)
	print(model.summary())

	global graph
	graph = tf.get_default_graph()

	# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	# callbacks_list = [checkpoint]

	# do 1 epoch for new data
	# fit network
	model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False) #, callbacks=callbacks_list
	# save model
	model.save(model_path)
	print("model saved !!!")

	# forecast
	to_predict = y
	act_val = scaler.inverse_transform(to_predict)
	print("Predict: ", act_val)

	with graph.as_default():  
		to_predict = to_predict.reshape((1, timesteps, features))

		yhat = model.predict(to_predict)
		inv_pred = scaler.inverse_transform(yhat)
		act_vals = scaler.inverse_transform(reframed)
		act_vals = act_vals[:,1]

		print("next ", inv_pred)

		hours = 48
		aa=[x for x in range(hours)]
		ab=[x for x in range(hours+1)]
		pred_to_plot = [None for _ in range(hours+1)]
		pred_to_plot[-1] = inv_pred[0][0]


		plt.plot(ab, pred_to_plot, 'r,' ,marker='.', label="prediction")
		plt.plot(aa, act_vals[-hours:], 'b', marker='.', label="actual")
		plt.ylabel('Energy (mWh)', size=15)
		plt.xlabel('Past Hours', size=15)
		plt.legend(fontsize=15)

		fig = plt.gcf()
		buf = io.BytesIO()
		fig.savefig(buf, format='png')
		buf.seek(0)
		string = base64.b64encode(buf.read())
		# io = StringIO()
		# fig.savefig(io, format='png')
		# string = base64.encodestring(io.getvalue())

		uri = 'data:image/png;base64,' + urllib.parse.quote(string)
		plt.clf()
		# end 
	clear_session()

	return render(request, 'result.html', {'prediction': inv_pred[0][0], 'act_val': act_val[0], 'fig': uri })

# convert data
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	dff = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(dff.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(dff.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg







