# region Import libraries
import src.utils._helpers as helpers
from src._parameters import Parameters, EvalParameters
from src.classes.Model import Model
import spacy
import os
import numpy as np
import json
# endregion

if __name__ == "__main__":

    np.random.seed(42)

    parameters = Parameters()
    eval_parameters = EvalParameters()

    print("Loading spacy...")
    nlp = spacy.load(parameters.spacy_model)

    print("Loading embeddings...")
    embeddings_model, embeddings_vocab, embeddings_dim = helpers.load_embeddings(parameters.embeddings_path)

    model_name = eval_parameters.model_name
    with open(os.path.join(eval_parameters.model_opts_path, model_name + ".json")) as json_file:
        opts = json.load(json_file)

    opts["load_weights"] = "True"
    opts["weights_path"] = os.path.join(eval_parameters.model_h5_path, model_name + ".h5")

    model = Model(None, None, opts)
    model.build_model()

    print("Prediction started...")
    for root, _, files in os.walk(eval_parameters.eval_unlabelled_path):
        for filename in files:
            if filename.startswith("task_1"):
                doc = os.path.join(root, filename)
                lines = open(doc, "r").readlines()
                for line in lines:
                    line = line.replace("\n", "")
                    split_line = line.split(" ")
                    sent_matrix = helpers.vectorise_sent(line, nlp, embeddings_vocab, embeddings_model, embeddings_dim)
                    pred = model.model.predict(sent_matrix)[0][0]
                    if pred > eval_parameters.prediction_threshold:
                        pred = 1
                    else:
                        pred = 0
                    with open(os.path.join(eval_parameters.eval_labelled_path, filename), "a+") as output:
                        print('{}\t"{}"\n'.format(line, pred), file=output)
            print("File is written into " + os.path.join(eval_parameters.eval_labelled_path, filename))



#
# results = []
# file_names = os.listdir("./data/dataset/eval")
# for file_name in file_names:
#     full_path = "./data/dataset/eval/" + file_name
#     full_path_labeled = "./data/dataset/eval_labeled/" + file_name
#     f = open(full_path, "r")
#     f_labeled = open(full_path_labeled, "a+")
#     line = f.readline()
#     while line:
#         line = line.replace("\n", "")
#         split_line = line.split(" ")
#         sent_matrix = vectorize_sent(line, embeddings_vocab, embeddings_model, embeddings_dim)
#         """
#         if len(split_line) < 30:
#             result = cblstm_model.predict(sent_matrix)[0][0]
#         else:
#         """
#         result = blstm_model.predict(sent_matrix)[0][0]
#         if 0.5 > result > 0.45:
#             #result = 1
#             written_line = '{}\t"{}"\n'.format(line, result)
#             f_labeled.write(written_line)
#         else:
#             result = 0
#         #written_line = '{}\t"{}"\n'.format(line, result)
#         #f_labeled.write(written_line)
#         results.append(result)
#         line = f.readline()
#     f.close()
#     f_labeled.close()
#
# print(results)
# print(len(results))
# print(results.count(1))
# print(results.count(0))