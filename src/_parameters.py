class Parameters:
	def __init__(self):
		# Constant params
		self.embeddings_path = "./data/embeddings/GoogleNews-vectors-negative300.bin"
		self.train_dataset_path = "./data/dataset/all/train"
		self.val_dataset_path = "./data/dataset/all/val"
		self.test_dataset_path = "./data/dataset/all/test"
		self.save_model_opts_path = "./data/models/opts"
		self.save_model_h5_path = "./data/models/h5"

		# Embedding params
		self.maxlen = 134
		self.idlen = 46

		# Spacy params
		self.spacy_model = "en_core_web_lg"

		# Model params
		self.save_model = True
		self.model_type = "blstm"
		self.epochs = 10
		self.batch_size = 128
		self.activate_attention = True
		self.dropout = 0.4

		# CNN params
		self.kernel_size = 3
		self.filters = 96
		self.pool_size = 2
		self.strides = 1

		# LSTM params
		self.lstm_units = 300
