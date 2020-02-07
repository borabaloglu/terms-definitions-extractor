import spacy
from sklearn.model_selection import train_test_split
import utils._helpers as helpers
from classes.Dataset import Dataset
from classes.Model import Model
from _parameters import Parameters
import pprint
pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":

	parameters = Parameters()

	print("Loading spacy")
	nlp = spacy.load(parameters.spacy_models[0])

	print("Loading dataset")
	dataset = Dataset(parameters.dataset_path)
	dataset.load_dataset()

	x, y = helpers.vectorize_dataset(dataset, nlp, parameters.embeddings_path)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=parameters.train_split_size, random_state=42)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=parameters.val_split_size, random_state=42)

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
								print("Train started with opts...")
								pp.pprint(opts)
								model = Model(x_train, y_train, opts)
								model.build_model()
								model.fit_model(epochs=parameters.epochs, batch_size=parameters.batch_size, validation_data=(x_val, y_val))
								precision, recall, f1 = model.calculate_metrics(x_test, y_test)
								print(precision, recall, f1)
						else:
							print("Train started with opts...")
							pp.pprint(opts)
							model = Model(x_train, y_train, opts)
							model.build_model()
							model.fit_model(epochs=parameters.epochs, batch_size=parameters.batch_size, validation_data=(x_val, y_val))
							precision, recall, f1 = model.calculate_metrics(x_test, y_test)
							print(precision, recall, f1)
