import spacy
import os
from keras import Model as KerasModel
from keras.layers import Dense, Bidirectional, Dropout, Conv1D, MaxPool1D, LSTM, Input
from src.utils._attention import attention_3d_block
import numpy as np
import src.utils._helpers as helpers

print("Loading spacy...")
nlp = spacy.load("en_core_web_lg")


def pad_words(tokens, maxlen, append_tuple=False):
    if len(tokens) > maxlen:
        return tokens[:maxlen]
    else:
        dif = maxlen - len(tokens)
        for _ in range(dif):
            if not append_tuple:
                tokens.append("UNK")
            else:
                tokens.append(("UNK", "UNK"))
        return tokens


def vectorize_sent(sent, embeddings_vocab, embeddings_model, embeddings_dim):
    tokens = [tok.orth_ for tok in nlp(sent.lower())]
    sent_matrix = []
    for token in pad_words(tokens, 134, append_tuple=False):
        if token in embeddings_vocab:
            vec = np.concatenate(
                [embeddings_model[token], np.zeros(46 + 1)])
            sent_matrix.append(vec)
        else:
            sent_matrix.append(
                np.zeros(embeddings_dim + 46 + 1))
    sent_matrix = np.array(sent_matrix)
    return np.array([sent_matrix])


def build_model_blstm():
    inputs = Input(shape=(134, 347))
    lstm_out = Bidirectional(LSTM(units=300, return_sequences=True))(inputs)
    dropout_out = Dropout(0.4)(lstm_out)
    attention_out = attention_3d_block(dropout_out)
    output = Dense(1, activation="sigmoid")(attention_out)
    model = KerasModel(inputs=[inputs], outputs=[output])
    model.load_weights("./data/eval_models/model_0.737037037037037.h5")
    model.compile(loss="binary_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])
    return model


def build_model_cblstm():
    inputs = Input(shape=(134, 347))
    lstm_out = Bidirectional(LSTM(units=300, return_sequences=True))(inputs)
    dropout_out = Dropout(0.4)(lstm_out)
    cnn_out = Conv1D(
        filters=128,
        kernel_size=5,
        strides=1,
        padding="valid",
        activation="relu",
    )(dropout_out)
    max_pool_out = MaxPool1D(pool_size=2)(cnn_out)
    dropout_out = Dropout(0.4)(max_pool_out)
    attention_out = attention_3d_block(dropout_out)
    output = Dense(1, activation="sigmoid")(attention_out)
    model = KerasModel(inputs=[inputs], outputs=[output])
    model.load_weights("./data/eval_models/model_0.7254901960784315.h5")
    model.compile(loss="binary_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])
    return model


blstm_model = build_model_blstm()
cblstm_model = build_model_cblstm()

print("Loading embeddings")
embeddings_model, embeddings_vocab, embeddings_dim = helpers.load_embeddings("./data/embeddings/GoogleNews-vectors-negative300.bin")

results = []
file_names = os.listdir("./data/dataset/eval")
for file_name in file_names:
    full_path = "./data/dataset/eval/" + file_name
    full_path_labeled = "./data/dataset/eval_labeled/" + file_name
    f = open(full_path, "r")
    f_labeled = open(full_path_labeled, "a+")
    line = f.readline()
    while line:
        line = line.replace("\n", "")
        split_line = line.split(" ")
        sent_matrix = vectorize_sent(line, embeddings_vocab, embeddings_model, embeddings_dim)
        """
        if len(split_line) < 30:
            result = cblstm_model.predict(sent_matrix)[0][0]
        else:
        """
        result = blstm_model.predict(sent_matrix)[0][0]
        if 0.5 > result > 0.45:
            #result = 1
            written_line = '{}\t"{}"\n'.format(line, result)
            f_labeled.write(written_line)
        else:
            result = 0
        #written_line = '{}\t"{}"\n'.format(line, result)
        #f_labeled.write(written_line)
        results.append(result)
        line = f.readline()
    f.close()
    f_labeled.close()

print(results)
print(len(results))
print(results.count(1))
print(results.count(0))