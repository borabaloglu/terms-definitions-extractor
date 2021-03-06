class Parameters:
	def __init__(self):
		# Constant params
		self.embeddings_path = "./data/embeddings/GoogleNews-vectors-negative300.bin"
		self.train_dataset_path = "./data/dataset/all/train/"
		self.val_dataset_path = "./data/dataset/all/val/"
		self.test_dataset_path = "./data/dataset/all/test/"
		self.model_opts_path = "./data/models/opts/"
		self.model_h5_path = "./data/models/h5/"

		# Embedding params
		self.maxlen = 134
		self.idlen = 46
		self.input_dims = (134, 347)

		# Spacy params
		self.spacy_model = "en_core_web_lg"

		# Model params
		self.save_model = True
		self.model_type = "blstm"
		self.epochs = 1
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


class EvalParameters:
	def __init__(self):
		self.model_name = ""
		self.eval_unlabelled_path = "./data/eval/unlabelled"
		self.eval_labelled_path = "./data/eval/labelled"
		self.model_opts_path = "./data/eval/models/opts"
		self.model_h5_path = "./data/eval/models/h5"
		self.prediction_threshold = 0.6
