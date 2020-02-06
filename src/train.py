import spacy
from sklearn.model_selection import train_test_split
import src.utils._helpers as helpers
from src.classes.Dataset import Dataset
from src.classes.Model import Model
from src._parameters import Parameters
import pprint
pp = pprint.PrettyPrinter(indent=4)

SPACY_MODEL = "en_core_web_sm"
EMBEDDINGS_PATH = "./data/embeddings/GoogleNews-vectors-negative300.bin"
DATASET_PATH = "./data/dataset/deftcorpus"
MODEL_TYPE = "cnn"

EPOCHS = 100
BATCH_SIZE = 128

if __name__ == "__main__":

	parameters = Parameters()

	print("Loading spacy")
	nlp = spacy.load(SPACY_MODEL)

	print("Loading dataset")
	dataset = Dataset(DATASET_PATH)
	dataset.load_dataset()

	x, y = helpers.vectorize_dataset(dataset, nlp)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

	for model_type in parameters.model_types:
		for kernel_size in parameters.kernel_sizes:
			for filter_size in parameters.filters:
				for pool_size in parameters.pool_sizes:
					for stride in parameters.strides:
						opts = {
							"model_type": model_type,
							"kernel_size": kernel_size,
							"filters": filter_size,
							"pool_size": pool_size,
							"strides": stride,
							"lstm_units": 0
						}
						if model_type == "cblstm":
							for lstm_unit in parameters.lstm_units:
								opts["lstm_units"] = lstm_unit
								model = Model(x_train, y_train, opts)
								model.build_model()
								model.fit_model(epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))
								precision, recall, f1 = model.calculate_metrics(x_test, y_test)
								pp.pprint(opts)
								print(precision, recall, f1)
						else:
							model = Model(x_train, y_train, opts)
							model.build_model()
							model.fit_model(epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))
							precision, recall, f1 = model.calculate_metrics(x_test, y_test)
							pp.pprint(opts)
							print(precision, recall, f1)
