import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score


class Metrics(Callback):
    def __init__(self):
        super().__init__()
        self.f1_values = []
        self.recall_values = []
        self.precision_values = []

    def on_train_begin(self, logs=None):
        self.f1_values = []
        self.recall_values = []
        self.precision_values = []

    def on_epoch_end(self, epoch, logs=None):
        preds = np.array([i[0].round() for i in self.model.predict(self.validation_data[0])])
        targets = self.validation_data[1]
        precision = precision_score(targets, preds)
        recall = recall_score(targets, preds)
        f1 = f1_score(targets, preds)
        self.precision_values.append(precision)
        self.recall_values.append(recall)
        self.f1_values.append(f1)
        print("precision: {}\trecall: {}\tf1: {}".format(precision, recall, f1))
        return


metrics = Metrics()
