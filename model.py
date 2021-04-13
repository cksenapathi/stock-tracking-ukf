# import numpy as np
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Reshape
from tensorflow.keras.models import Sequential


class Model:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=4, input_shape=(4, 4), activation='relu'))
        self.model.add(MaxPooling1D(1))
        self.model.add(Dropout(.2))

        self.model.add(Dense(32))
        self.model.add(Dense(16))

        self.model.add(Reshape(target_shape=(4, 4)))

    def predict(self, x, training=False):
        return self.model(x, training=training)

# print(model.get_weights())
# model.fit(x=x_train, y=y_train, epochs=3, batch_size=1, verbose=0)

# test_acc = model.evaluate(x_test, y_test, verbose=2, batch_size=1)

# print(f'\nTest Accuracy:{test_acc}')
