# region Import libraries
import spacy
import src.utils._helpers as helpers
from src.classes.Dataset import Dataset
from src.classes.Model import Model
from src._parameters import Parameters
import os
import numpy as np
import json
import pprint
import uuid
pp = pprint.PrettyPrinter(indent=4)
# endregion

if __name__ == "__main__":

	np.random.seed(42)

	parameters = Parameters()

	print("Loading spacy...")
	nlp = spacy.load(parameters.spacy_model)

	print("Loading train dataset...")
	train_dataset = Dataset(parameters.train_dataset_path)
	train_dataset.load_dataset()

	print("Loading validation dataset...")
	val_dataset = Dataset(parameters.val_dataset_path)
	val_dataset.load_dataset()

	print("Loading test dataset...")
	test_dataset = Dataset(parameters.test_dataset_path)
	test_dataset.load_dataset()

	print("Loading embeddings...")
	embeddings_model, embeddings_vocab, embeddings_dim = helpers.load_embeddings(parameters.embeddings_path)
	embeddings = {
		"model": embeddings_model,
		"vocab": embeddings_model,
		"dim": embeddings_dim
	}

	print("Vectorising train dataset...")
	x_train, y_train = helpers.vectorise_dataset(train_dataset, nlp, embeddings, parameters.maxlen, parameters.idlen)

	print("Vectorising validation dataset...")
	x_val, y_val = helpers.vectorise_dataset(val_dataset, nlp, embeddings, parameters.maxlen, parameters.idlen)

	print("Vectorising test dataset...")
	x_test, y_test = helpers.vectorise_dataset(test_dataset, nlp, embeddings, parameters.maxlen, parameters.idlen)

	opts = {
		"model_type": parameters.model_type,
		"activate_attention": parameters.activate_attention,
		"dropout": parameters.dropout,
		"input_dims": parameters.input_dims
	}
	if parameters.model_type == "cnn":
		opts["kernel_size"] = parameters.kernel_size
		opts["filters"] = parameters.filters
		opts["pool_size"] = parameters.pool_size
		opts["strides"] = parameters.strides
	elif parameters.model_type == "blstm":
		opts["lstm_units"] = parameters.lstm_units
	else:
		pass

	print("== OPTS ==")
	pp.pprint(opts)

	model = Model(x_train, y_train, opts)
	model.build_model()
	model.fit_model(epochs=parameters.epochs, batch_size=parameters.batch_size, validation_data=(x_val, y_val))

	history = model.model.history.history
	opts["history"] = {
		"val_loss": str(history["val_loss"]),
		"val_accuracy": str(history["val_accuracy"]),
		"loss": str(history["loss"]),
		"accuracy": str(history["accuracy"])
	}
	pp.pprint(opts["history"])

	precision, recall, f1 = model.calculate_metrics(x_test, y_test)
	opts["metrics"] = {
		"precision": str(precision),
		"recall": str(recall),
		"f1": str(f1)
	}

	if parameters.save_model:
		unique_id = str(uuid.uuid1())
		file_name = str(f1) + "_" + unique_id
		opts_path = os.path.join(parameters.model_opts_path, file_name + ".json")
		h5_path = os.path.join(parameters.model_h5_path, file_name + ".h5")
		with open(opts_path, "w") as output:
			json.dump(opts, output, sort_keys=True, indent=4)
		model.model.save(h5_path)
		print("Your model opts is saved into " + opts_path)
		print("Your model is saved into " + h5_path)
