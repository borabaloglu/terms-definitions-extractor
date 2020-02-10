import gensim
import sys
import numpy as np
from sklearn.utils import shuffle


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
                sys.exit("Couldn't load embeddings")
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


def vectorize_dataset(datasets, nlp, embeddings):
    embeddings_model = embeddings["model"]
    embeddings_vocab = embeddings["vocab"]
    embeddings_dim = embeddings["dim"]

    print("Getting maxlen")
    maxlen, deps2ids = get_maxlen(datasets[0], nlp)
    ids2deps = dict([(idx, dep) for dep, idx in deps2ids.items()])

    print("Vectorizing dataset")
    x_train = []
    x_val = []
    x_test = []
    for i in range(len(datasets)):
        dataset = datasets[i]
        for idx, sent in enumerate(dataset.instances):
            if idx % 1000 == 0:
                print("Done " + str(idx) + " of " + str(len(dataset.instances)))
            tokens = [tok.orth_ for tok in nlp(sent.lower())]
            sent_matrix = []
            for token in pad_words(tokens, maxlen, append_tuple=False):
                if token in embeddings_vocab:
                    vec = np.concatenate(
                        [embeddings_model[token], np.zeros(len(ids2deps) + 1)])
                    sent_matrix.append(vec)
                else:
                    sent_matrix.append(
                        np.zeros(embeddings_dim + len(ids2deps) + 1))
            sent_matrix = np.array(sent_matrix)
            if i == 0:
                x_train.append(sent_matrix)
            elif i == 1:
                x_val.append(sent_matrix)
            else:
                x_test.append(sent_matrix)
    x_train = np.array(x_train)
    y_train = np.array(datasets[0].labels)
    x_val = np.array(x_val)
    y_val = np.array(datasets[1].labels)
    x_test = np.array(x_test)
    y_test = np.array(datasets[2].labels)
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_val, y_val = shuffle(x_val, y_val, random_state=42)
    x_test, y_test = shuffle(x_test, y_test, random_state=42)
    return x_train, y_train, x_val, y_val, x_test, y_test
