from ukf import UKF
import json
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from math import prod

# Uninstall old CUDA version, basically anything with a 10, and install
# all 11s


def gather_data(ticker='AAPL', back_tracking_count=4):
    table = json.dumps((requests.get('https://finnhub.io/api/v1/stock/candle?symbol={}&resolution=1&from=1594639800&'
                                     'to=1618105881&token=br78u2frh5r9l4n3ntgg'.format(ticker))).json())
    table = pd.read_json(table)
    table.pop('s')
    table.pop('t')
    table.pop('v')
    mat = table.to_numpy()

    x = np.array([np.zeros(shape=(4, 4))])
    y = np.array([np.zeros(shape=(4, 4))])

    k = back_tracking_count
    for i in range(len(mat)-k-1):
        x = np.append(x, np.array([mat[i:i+k, :]]), axis=0)
        y = np.append(y, np.array([mat[i+1:i+1+k, :]]), axis=0)
    x = np.delete(x, 0, axis=0)
    y = np.delete(y, 0, axis=0)
    return x, y


def split_data(x, y, test_percent=.25):
    test_len = int(len(x) * test_percent)
    shuffle = np.random.permutation(len(x))
    x = x[shuffle]
    y = y[shuffle]
    x_test = x[-test_len:, :, :]
    y_test = y[-test_len:, :, :]

    x_train = x[:-test_len, :, :]
    y_train = y[:-test_len, :, :]

    return x_train, y_train[:, 0, :], x_test, y_test[:, 0, :]


class Model:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=4, input_shape=(4, 4), activation='relu'))
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
        for j, weight in enumerate(weights):
            self.model.get_layer(index=j).set_weights(weights=weight)
        return


class Optimizer:
    def __init__(self):
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
        self._last_grad = []

    def loss(self, model, x, y, tra=False):
        y_ = model.predict(x, training=tra)
        return self.loss_object(y_true=y, y_pred=y_)

    def grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, inputs, targets, tra=True)
        curr_grad = tape.gradient(loss_value, model.get_trainable_variables())
        self._last_grad = curr_grad
        return loss_value, curr_grad

    def get_weight_cov(self):
        weight_changes = np.array([])
        for i in self._last_grad:
            weight_changes = np.append(weight_changes, np.abs(i.numpy()))
        return np.outer(weight_changes - np.mean(weight_changes), weight_changes - np.mean(weight_changes))


def measurement_noise(predict, real):
    return np.outer((real - predict), (real-predict))


def main():
    x, y = gather_data()
    # print('{} {}'.format(x.shape, type(x)))
    # print('{} {}'.format(y.shape, type(y)))
    x_train, y_train, x_test, y_test = split_data(x, y)
    model = Model()
    opt = Optimizer()
    print(type(model.get_model()))
    epochs = 4
    # error = np.array([])
    # grad = np.array([])
    for epoch in range(epochs):
        for i in range(len(x_train)):
            loss_value, grad = opt.grad(model, np.array([x_train[i, :, :]]), np.array([y_train[i, :]]))
            opt.optimizer.apply_gradients((zip(grad, model.get_trainable_variables())))
    #         if epoch*len(x_train) + i % 250 == 0:
    #             error = np.append(error, loss_value.numpy())
    # plt.figure('MSE Loss over All Training')
    # plt.plot(error, 'b-')
    # plt.grid(axis='both', b=True, color='k', linewidth=.2)
    # plt.show()

    # print('finished training, beginning testing')
    # ukf = UKF(mean=x_test[0, :, :].reshape(16), covar=np )
    # for i in range(len(y_test)):
    #     y_predict = model.predict(x_test[i, :, :], training=False)
    #     # Model takes in and spits out (4,4) arrays

    ''' On the training set, it needs to be done sequentially; 
    We initialize the model, train it and with testing, it needs to:
        transition the weights first
        calculate the gradient and add as process noise
        then it needs to output the prediction
        then the real needs to be combined
        weight transition is done with  '''
    state, shape = model.get_weight_state()
    ukf = UKF(mean=state, covariance=opt.get_weight_cov(), model=model)
    m_noise = np.zeros(4)
    sigma, mean_weights, covar_weights = ukf.calc_sigma_points(state, opt.get_weight_cov())
    predicted_y = np.array([])
    for i in range(len(x_test)):
        # Predicts output based on current weight state
        predict_output, predict_output_mean, predict_output_covar = ukf.output(shapes=shape, x=x_test[i, :, :],
                                                                               sigma=sigma, mean_weights=mean_weights,
                                                                               covar_weights=covar_weights,
                                                                               measurement_noise=m_noise)
        # Get and insert real values
        predicted_y = np.append(predicted_y, predict_output_mean)
        ukf.update(y_test[i, :], sigma, mean_weights, covar_weights, predict_output, predict_output_mean,
                   predict_output_covar)
        m_noise = y_test[i, :] - predict_output_mean
        _, grad = opt.grad(model, np.array([x_test[i, :, :]]), np.array([y_test[i, :]]))
        mean, covar = ukf.get_state()
        sigma, mean_weights, covar_weights = ukf.calc_sigma_points(mean, covar)

    predicted_y = predicted_y.reshape((4, -1))
    fig, ((p1, p2), (p3, p4)) = plt.subplots(2, 2)
    # plt.figure('Predicted Values')
    # plt.clf()
    # plt.plot(domain, predicted_y[:, 0, :])
    # plt.grid(axis='both', b=True, color='k', linewidth=.2)
    print(y_test.shape)
    print(predicted_y.shape)
    p1.plot(y_test[:len(y_test)//10, 0], label='True')
    p1.plot(predicted_y[:len(y_test)//10, 0], label='Predicted')
    p1.set_ylim(bottom=100, top=150)
    p1.legend()

    p2.plot(y_test[:len(y_test)//10, 1], label='True')
    p2.plot(predicted_y[:len(y_test)//10, 1], label='Predicted')
    p2.set_ylim(bottom=100, top=150)
    p2.legend()

    p3.plot(y_test[:len(y_test)//10, 2], label='True')
    p3.plot(predicted_y[:len(y_test)//10, 2], label='Predicted')
    p3.set_ylim(bottom=100, top=150)
    p3.legend()

    p4.plot(y_test[:len(y_test)//10, 3], label='True')
    p4.plot(predicted_y[:len(y_test)//10, 3], label='Predicted')
    p4.set_ylim(bottom=100, top=150)
    p4.legend()

    plt.show()

    plt.figure('True Values')
    plt.plot(y_test[:len(y_test)//10, 0], label='open')
    plt.plot(y_test[:len(y_test)//10, 1], label='high')
    plt.plot(y_test[:len(y_test)//10, 2], label='low')
    plt.plot(y_test[:len(y_test)//10, 3], label='close')
    plt.show()

    # print(f'grad Type: {type(grad)}')
    # print(f'Shape of grad:{np.array(grad).shape}')# Grad:{grad}\n\n')
    # print(f'Model Parameters Type: {type(model.get_trainable_variables())}')
    # vars = model.get_trainable_variables()
    # opt.optimizer.apply_gradients(zip(grad, model.get_trainable_variables()))
    # arr = np.array([])
    # for i in range(len(vars)):
    #     delta = vars[i].numpy() - grad[i].numpy()
    #     # print(np.sum(np.abs(delta)))
    #     # print(np.sum(np.abs(grad[i].numpy())))
    #     arr = np.append(arr, grad[i].numpy())
    #
    # print(arr.shape)
    # ukf = UKF()

    # for i in range(len(grad)):
    #     print(grad[i].name)


if __name__ == '__main__':
    main()
