import spacy
from sklearn.model_selection import train_test_split
import src.utils._helpers as helpers
from src.classes.Dataset import Dataset
from src.classes.Model import Model
from src._parameters import Parameters
import pprint
pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":

	parameters = Parameters()

	print("Loading spacy")
	nlp = spacy.load(parameters.spacy_models[0])

	print("Loading dataset")
	train_dataset = Dataset(parameters.train_dataset_path)
	train_dataset.load_dataset()

	print("Loading validation dataset")
	val_dataset = Dataset(parameters.val_dataset_path)
	val_dataset.load_dataset()

	print("Loading test dataset")
	test_dataset = Dataset(parameters.test_dataset_path)
	test_dataset.load_dataset()

	print("Loading embeddings")
	embeddings_model, embeddings_vocab, embeddings_dim = helpers.load_embeddings(parameters.embeddings_path)
	embeddings = {
		"model": embeddings_model,
		"vocab": embeddings_model,
		"dim": embeddings_dim
	}

	datasets = [train_dataset, val_dataset, test_dataset]
	x_train, y_train, x_val, y_val, x_test, y_test = helpers.vectorize_dataset(datasets, nlp, embeddings)

	max_f1 = parameters.f1_threshold
	for model_type in parameters.model_types:
		for kernel_size in parameters.kernel_sizes:
			for filter_size in parameters.filters:
				for pool_size in parameters.pool_sizes:
					for stride in parameters.strides:
						for attention in parameters.activate_attention:
							opts = {
								"model_type": model_type,
								"kernel_size": kernel_size,
								"filters": filter_size,
								"pool_size": pool_size,
								"strides": stride,
								"lstm_units": 0,
								"activate_attention": attention
							}
							if model_type == "blstm":
								for lstm_unit in parameters.lstm_units:
									opts["lstm_units"] = lstm_unit
									print("Train started with opts...")
									pp.pprint(opts)
									model = Model(x_train, y_train, opts)
									model.build_model()
									model.fit_model(epochs=parameters.epochs, batch_size=parameters.batch_size, validation_data=(x_val, y_val))
									precision, recall, f1 = model.calculate_metrics(x_test, y_test)
									print(precision, recall, f1)
									if f1 > max_f1:
										model.model.save("./data/models/model_" + str(f1) + ".h5")
										f = open("./data/models/model_" + str(f1) + ".json")
										opts["results"] = {
											"precision": precision,
											"recall": recall,
											"f1": f1
										}
										opts["history"] = model.model.history
										f.write(str(opts))
										f.close()
										max_f1 = f1
							else:
								print("Train started with opts...")
								pp.pprint(opts)
								model = Model(x_train, y_train, opts)
								model.build_model()
								model.fit_model(epochs=parameters.epochs, batch_size=parameters.batch_size, validation_data=(x_val, y_val))
								precision, recall, f1 = model.calculate_metrics(x_test, y_test)
								print(precision, recall, f1)
								if f1 > max_f1:
									model.model.save("./data/models/model_" + str(f1) + ".h5")
									f = open("./data/models/model_" + str(f1) + ".json")
									opts["results"] = {
										"precision": precision,
										"recall": recall,
										"f1": f1
									}
									opts["history"] = model.model.history
									f.write(str(opts))
									f.close()
									max_f1 = f1
