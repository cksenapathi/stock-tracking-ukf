import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Dense
from math import prod
from tensorflow.keras.models import load_model


class Model:
    def __init__(self, filepath=None):
        if filepath is not None:
            self.model = load_model(filepath=filepath)
        else:
            self.model = Sequential()
            self.model.add(Conv1D(filters=16, kernel_size=4, input_shape=(4, 4), activation='relu'))
            self.model.add(MaxPooling1D(1))
            self.model.add(Dropout(.2))

            self.model.add(Dense(32))
            self.model.add(Dense(16))
            self.model.add(Dense(4))

    def predict(self, x, training=False):
        return self.model(x, training=training)

    def get_trainable_variables(self):
        return self.model.trainable_variables

    def get_model(self):
        return self.model

    def summary(self):
        return self.model.summary()

    def get_weight_state(self):
        # Current idea is to pass the single vector and also the original shapes of the layers, so that when all of it
        # is done, i can reshape them back and then update the gradient with them
        weights = self.model.trainable_variables
        shapes = []
        ret = np.array([])
        for i in weights:
            ret = np.append(ret, i.numpy())
            shapes.append(i.numpy().shape)
        return ret, shapes

    def set_weights(self, shapes, i):
        weights = []
        for j in shapes:
            length = prod(j)
            layer, i = i[:length], i[length:]
            weights.append(layer.reshape(j))
        self.model.set_weights(weights=weights)
        return
