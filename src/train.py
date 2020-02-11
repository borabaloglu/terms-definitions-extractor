import spacy
import src.utils._helpers as helpers
from src.classes.Dataset import Dataset
from src.classes.Model import Model
from src._parameters import Parameters
import pprint
pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":

	parameters = Parameters()

	print("Loading spacy")
	nlp = spacy.load(parameters.spacy_model)

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

	if parameters.model_type == "cnn":
		for kernel_size in parameters.kernel_sizes:
			for filter_size in parameters.filters:
				for pool_size in parameters.pool_sizes:
					for stride in parameters.strides:
						for attention in parameters.activate_attention:
							opts = {
								"model_type": parameters.model_type,
								"kernel_size": kernel_size,
								"filters": filter_size,
								"pool_size": pool_size,
								"strides": stride,
								"activate_attention": attention
							}
							print("Train started with opts...")
							pp.pprint(opts)
							model = Model(x_train, y_train, opts)
							model.build_model()
							model.fit_model(epochs=parameters.epochs, batch_size=parameters.batch_size, validation_data=(x_val, y_val))
							precision, recall, f1 = model.calculate_metrics(x_test, y_test)
							print(precision, recall, f1)
							model.model.save("./data/models/model_" + str(f1) + ".h5")
							f = open("./data/models/model_" + str(f1) + ".json", "w")
							opts["results"] = {
								"precision": precision,
								"recall": recall,
								"f1": f1
							}
							history = model.model.history.history
							opts["history"] = {
								"val_loss": history["val_loss"],
								"val_accuracy": history["val_accuracy"],
								"loss": history["loss"],
								"accuracy": history["accuracy"]
							}
							f.write(str(opts))
							f.close()
	else:
		for lstm_unit in parameters.lstm_units:
			for attention in parameters.activate_attention:
				opts = {
					"model_type": parameters.model_type,
					"lstm_units": lstm_unit,
					"activate_attention": attention
				}
				print("Train started with opts...")
				pp.pprint(opts)
				model = Model(x_train, y_train, opts)
				model.build_model()
				model.fit_model(epochs=parameters.epochs, batch_size=parameters.batch_size, validation_data=(x_val, y_val))
				precision, recall, f1 = model.calculate_metrics(x_test, y_test)
				print(precision, recall, f1)
				model.model.save("./data/models/model_" + str(f1) + ".h5")
				f = open("./data/models/model_" + str(f1) + ".json", "w")
				opts["results"] = {
					"precision": precision,
					"recall": recall,
					"f1": f1
				}
				history = model.model.history.history
				opts["history"] = {
					"val_loss": history["val_loss"],
					"val_accuracy": history["val_accuracy"],
					"loss": history["loss"],
					"accuracy": history["accuracy"]
				}
				f.write(str(opts))
				f.close()
