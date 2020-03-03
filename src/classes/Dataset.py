# region Import libraries
import numpy as np
import os
# endregion


class Dataset(object):
	def __init__(self, path):
		self.path = path
		self.instances = []
		self.labels = []

	def load_dataset(self):
		for root, _, files in os.walk(self.path):
			for filename in files:
				if filename.startswith("task_1"):
					doc = os.path.join(root, filename)
					lines = open(doc, "r").readlines()
					for idx, line in enumerate(lines):
						split_line = line.split("\t")
						sentence = split_line[0].replace("\"", "")
						label = split_line[1].replace("\n", "")
						label = label.replace("\"", "")
						label = int(label)
						self.instances.append(sentence)
						self.labels.append(label)
		self.labels = np.array(self.labels)
