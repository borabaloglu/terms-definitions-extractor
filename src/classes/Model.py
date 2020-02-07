import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Bidirectional, Dropout, Conv1D, MaxPooling1D, Embedding, Add, Flatten, LSTM
from keras.activations import tanh
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class Model(object):
	def __init__(self, x, y, opts):
		self.model = None
		self.x = x
		self.y = y
		self.model_type = opts["model_type"]
		self.activation_func = "sigmoid"
		self.loss_func = "binary_crossentropy"
		self.optimizer = "adam"
		self.metrics = ["accuracy"]
		self.cnn_opts = {
			"kernel_size": opts["kernel_size"],
			"filters": opts["filters"],
			"pool_size": opts["pool_size"],
			"strides": opts["strides"],
			"padding": "valid",
			"activation": "relu",
		}
		self.lstm_opts = {
			"units": opts["lstm_units"]
		}

	def build_model(self):
		model = Sequential()
		model.add(Conv1D(
			filters=self.cnn_opts["filters"],
			kernel_size=self.cnn_opts["kernel_size"],
			strides=self.cnn_opts["strides"],
			padding=self.cnn_opts["padding"],
			activation=self.cnn_opts["activation"],
			input_shape=(self.x.shape[1], self.x.shape[2])
		))
		model.add(MaxPooling1D(pool_size=self.cnn_opts["pool_size"]))
		if self.model_type == "cnn":
			model.add(Flatten())
			model.add(Dropout(0.5))
		elif self.model_type == "cblstm":
			model.add(Bidirectional(LSTM(units=self.lstm_opts["units"])))
			model.add(Dropout(0.5))
		else:
			pass
		model.add(Dense(1))
		model.add(Activation(self.activation_func))
		model.compile(
			loss=self.loss_func,
			optimizer=self.optimizer,
			metrics=self.metrics
		)
		self.model = model

	def fit_model(self, epochs, batch_size, validation_data):
		self.model.fit(self.x, self.y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=False)

	def calculate_metrics(self, x_test, y_test):
		preds = np.array([i[0] for i in self.model.predict_classes(x_test)])
		precision = precision_score(y_test, preds)
		recall = recall_score(y_test, preds)
		f1 = f1_score(y_test, preds)
		return precision, recall, f1
