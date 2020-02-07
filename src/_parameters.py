class Parameters:
	def __init__(self):
		# Constant params
		self.embeddings_path = "src/data/embeddings/GoogleNews-vectors-negative300.bin"
		self.dataset_path = "src/data/dataset/deftcorpus"

		# Spacy params
		self.spacy_models = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]

		# Model params
		self.model_types = ["cnn", "cblstm"]
		self.epochs = 20
		self.batch_size = 128
		self.train_split_size = 0.2
		self.val_split_size = 0.2

		# CNN params
		self.kernel_sizes = [3, 6, 9]
		self.filters = [50, 75, 100]
		self.pool_sizes = [2, 4, 6]
		self.strides = [1, 2, 3]

		# LSTM params
		self.lstm_units = [50, 75, 100]
