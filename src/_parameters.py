class Parameters:
	def __init__(self):
		# Constant params
		self.embeddings_path = "./data/embeddings/GoogleNews-vectors-negative300.bin"
		self.train_dataset_path = "./data/dataset/train"
		self.val_dataset_path = "./data/dataset/val"
		self.test_dataset_path = "./data/dataset/test"

		# Spacy params
		self.spacy_model = "en_core_web_sm"

		# Model params
		self.model_type = "cnn"
		self.epochs = 10
		self.batch_size = 64
		self.train_split_size = 0.2
		self.val_split_size = 0.2
		self.activate_attention = [True, False]

		# CNN params
		self.kernel_sizes = [3, 6, 9]
		self.filters = [96, 192, 256]
		self.pool_sizes = [2, 4]
		self.strides = [1, 2]

		# LSTM params
		self.lstm_units = [96, 192, 256]

		self.f1_threshold = 0.6
