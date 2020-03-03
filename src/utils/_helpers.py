# region Import libraries
import gensim
import sys
import numpy as np
from sklearn.utils import shuffle
# endregion


def load_embeddings(path):
    try:
        model = gensim.models.Word2Vec.load(path)
    except:
        try:
            model = gensim.models.KeyedVectors.load_word2vec_format(path)
        except:
            try:
                model = gensim.models.KeyedVectors.load_word2vec_format(
                    path, binary=True)
            except:
                sys.exit("Loading embeddings failed!")
    vocab = model.index2word
    dim = model.__getitem__(vocab[0]).shape[0]
    vocab = set(vocab)
    return model, vocab, dim


def get_maxlen(dataset, nlp):
    maxlen = 0
    maxlen_dep = 0
    deps2ids = {}
    dep_id = 0

    for idx, sent in enumerate(dataset.instances):
        if idx % 1000 == 0:
            print("Done " + str(idx) + " of " + str(len(dataset.instances)))
        try:
            sent_maxlen_dep = 0
            doc = nlp(sent)
            if len(doc) > maxlen:
                maxlen = len(doc)
            for token in doc:
                if token.dep_ not in deps2ids:
                    deps2ids[token.dep_] = dep_id
                    dep_id += 1
                for child in token.children:
                    if child.dep_ not in deps2ids:
                        deps2ids[child.dep_] = dep_id
                        dep_id += 1
                    sent_maxlen_dep += 1
            if sent_maxlen_dep > maxlen_dep:
                maxlen_dep = sent_maxlen_dep
        except UnicodeDecodeError:
            pass

    return max(maxlen, maxlen_dep), deps2ids


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


def vectorise_dataset(dataset, nlp, embeddings, maxlen, idlen):
    embeddings_model = embeddings["model"]
    embeddings_vocab = embeddings["vocab"]
    embeddings_dim = embeddings["dim"]

    x = []
    for idx, sent in enumerate(dataset.instances):
        if idx % 1000 == 0:
            print("Done " + str(idx) + " of " + str(len(dataset.instances)))
        tokens = [tok.orth_ for tok in nlp(sent.lower())]
        sent_matrix = []
        for token in pad_words(tokens, maxlen, append_tuple=False):
            if token in embeddings_vocab:
                vec = np.concatenate([embeddings_model[token], np.zeros(idlen + 1)])
                sent_matrix.append(vec)
            else:
                sent_matrix.append(np.zeros(embeddings_dim + idlen + 1))
        sent_matrix = np.array(sent_matrix)
        x.append(sent_matrix)
    x = np.array(x)
    y = np.array(dataset.labels)
    x, y = shuffle(x, y, random_state=42)
    return x, y


def vectorise_sent(sent, nlp, embeddings_vocab, embeddings_model, embeddings_dim):
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