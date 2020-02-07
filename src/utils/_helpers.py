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
				model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
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
		for i in range(dif):
			if not append_tuple:
				tokens.append("UNK")
			else:
				tokens.append(("UNK", "UNK"))
		return tokens


def vectorize_dataset(dataset, nlp, embeddings_path):
	print("Loading embeddings")
	embeddings_model, embeddings_vocab, embeddings_dim = load_embeddings(
		embeddings_path)

	print("Getting maxlen")
	maxlen, deps2ids = get_maxlen(dataset, nlp)
	ids2deps = dict([(idx, dep) for dep, idx in deps2ids.items()])

	print("Vectorizing dataset")
	x = []
	for idx, sent in enumerate(dataset.instances):
		if idx % 1000 == 0:
			print("Done " + str(idx) + " of " + str(len(dataset.instances)))
		tokens = [tok.orth_ for tok in nlp(sent.lower())]
		sent_matrix = []
		for token in pad_words(tokens, maxlen, append_tuple=False):
			if token in embeddings_vocab:
				vec = np.concatenate([embeddings_model[token], np.zeros(len(ids2deps) + 1)])
				sent_matrix.append(vec)
			else:
				sent_matrix.append(np.zeros(embeddings_dim + len(ids2deps) + 1))
		sent_matrix = np.array(sent_matrix)
		x.append(sent_matrix)

	x = np.array(x)

	"""
	x_word_pairs = []
	x_deps = []
	for idx, sent in enumerate(dataset.instances):
		if idx % 10 == 0:
			print("Done " + str(idx) + " of " + str(len(dataset.instances)))
		tokens = nlp(sent.lower())
		word_pairs = []
		dep_pairs = []
		for tok in tokens:
			for child in tok.children:
				word_pairs.append((tok.orth_, child.orth_))
				dep_pairs.append((tok.dep_, child.dep_))
		padded_wp = pad_words(word_pairs, maxlen, append_tuple=True)
		padded_deps = pad_words(dep_pairs, maxlen, append_tuple=True)
		dep_labels = [j for i,j in dep_pairs]
		avg_sent_matrix = []
		avg_label_sent_matrix = []
		for idx, word_pair in enumerate(word_pairs):
			head, modifier = word_pair[0], word_pair[1]

			if head in embeddings_vocab and not head == "UNK":
				head_vec = embeddings_model[head]
			else:
				head_vec = np.zeros(embeddings_dim)
			if modifier in embeddings_vocab and not modifier == "UNK":
				modifier_vec = embeddings_model[modifier]
			else:
				modifier_vec = np.zeros(embeddings_dim)

			avg = np.mean(np.array([head_vec, modifier_vec]), axis=0)

			if dep_labels[idx] != "UNK":
				dep_idx = deps2ids[dep_labels[idx]]
			else:
				dep_idx = -1

			dep_vec = np.zeros(len(deps2ids)+1)
			dep_vec[dep_idx] = 1
			avg_label_vec = np.concatenate([avg, dep_vec])
			avg_sent_matrix.append(np.concatenate([avg, np.zeros(len(deps2ids) + 1)]))
			avg_label_sent_matrix.append(avg_label_vec)
		wp = np.array(avg_sent_matrix)
		labs = np.array(avg_label_sent_matrix)
		x_word_pairs.append(wp)
		x_deps.append(labs)

	x_word_pairs = np.array(x_word_pairs)
	x_deps = np.array(x_deps)
	"""

	y = np.array(dataset.labels)
	return shuffle(x, y, random_state=0)
