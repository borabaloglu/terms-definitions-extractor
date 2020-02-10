from keras import Model as KerasModel
from keras.layers import Dense, Bidirectional, Dropout, Conv1D, MaxPool1D, Flatten, LSTM, Input
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from src.utils._attention import attention_3d_block


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
		self.activate_attention = opts["activate_attention"]
		if self.model_type == "cnn":
			self.cnn_opts = {
				"kernel_size": opts["kernel_size"],
				"filters": opts["filters"],
				"pool_size": opts["pool_size"],
				"strides": opts["strides"],
				"padding": "valid",
				"activation": "relu",
			}
		else:
			self.lstm_opts = {
				"units": opts["lstm_units"]
			}

	def build_model(self):
		inputs = Input(shape=(self.x.shape[1], self.x.shape[2]))
		if self.model_type == "cnn":
			cnn_out = Conv1D(
				filters=self.cnn_opts["filters"],
				kernel_size=self.cnn_opts["kernel_size"],
				strides=self.cnn_opts["strides"],
				padding=self.cnn_opts["padding"],
				activation=self.cnn_opts["activation"],
			)(inputs)
			max_pool_out = MaxPool1D(pool_size=self.cnn_opts["pool_size"])(cnn_out)
			dropout_out = Dropout(0.2)(max_pool_out)
		else:
			lstm_out = Bidirectional(LSTM(units=self.lstm_opts["units"], return_sequences=True))(inputs)
			dropout_out = Dropout(0.2)(lstm_out)
		if self.activate_attention:
			attention_out = attention_3d_block(dropout_out)
		else:
			attention_out = Flatten()(dropout_out)
		output = Dense(1, activation=self.activation_func)(attention_out)
		self.model = KerasModel(inputs=[inputs], outputs=[output])
		self.model.compile(loss=self.loss_func,
						   optimizer=self.optimizer,
						   metrics=self.metrics)

	def fit_model(self, epochs, batch_size, validation_data):
		self.model.fit(self.x, self.y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=True)

	def calculate_metrics(self, x_test, y_test):
		preds = np.array([i[0].round() for i in self.model.predict(x_test)])
		precision = precision_score(y_test, preds)
		recall = recall_score(y_test, preds)
		f1 = f1_score(y_test, preds)
		return precision, recall, f1
