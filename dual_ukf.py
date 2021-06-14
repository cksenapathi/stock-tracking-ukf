import numpy as np
import tensorflow as tf
import pandas as pd
from ukf import UKF
from model import Model
from optimizer import Optimizer
from utils import split_data, unzip, calc_covars
import matplotlib.pyplot as plt
from wt_ukf import WeightUKF
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from dyn_update import DynamicUpdate


def main():
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')

    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')

    x_val = np.load('x_val.npy')
    y_val = np.load('y_val.npy')

    # model1 = Model(filepath='open_model')
    # I'm going to train a model for 1 epoch, which I
    # will use to gather all the covariances
    # after the one epoch, I'll let the model run through
    # the other data and see how it performs
    # with less accurate guesses, I think the gradients
    # won't show signs of linear independence
    model1 = Sequential()
    model1.add(Dense(4))
    model1.add(Dense(8))
    model1.add(Dense(1))
    lr = .001

    opt = Optimizer(lr=lr)
    # history = 5
    # param_hist = 69
    # wt_data, grad_data, err_data = calc_covars(model1, opt, x_test[:, 0, :], y_test[:, 0], history, param_hist)
    price_data = x_train[-history:, 0, :].T
    price_mean = price_data[:, -1]
    price_cov = np.cov(price_data)
    values = np.zeros(len(y_test)+7)
    wts, shapes = model1.get_weight_state()
    values[0:4] = x_test[0, 0, :]
    values[4:7] = x_test[1:4,0,0]
    print(values[:8])

    price_ukf = UKF(mean=price_mean, covariance=price_cov, model=model1, shapes=shapes)
    wt_ukf = WeightUKF(mean=model1.get_weight_state()[0], cov=np.cov(wt_data),
                       model=model1, shapes=shapes, opt=opt)
    '''
        I need some ideas here
        I can start with an entirely untrained model and going through the data will eventually train it
        On the other hand I can train it one round, gather the various values and leave it running, and see what happens
        I'm not sure what's going to happen, but i need to fix it

    '''
    wt_update = DynamicUpdate(len(model1.get_weight_state()[0]))
    grad_update = DynamicUpdate(len(model1.get_weight_state()[0]))
    price_update = DynamicUpdate(4)
    err_update = DynamicUpdate(1)

    # What do I use to update
    wt_mean = model1.get_weight_state()[0]

    wt_mean, wt_cov = wt_update.update(wt_mean)
    price_mean, price_cov = price_update.update(values[:4])
    grad_mean = np.zeros((len(wt_mean), 1))
    grad_cov = np.zeros((len(wt_mean), len(wt_mean))) + .1 * np.eye(len(wt_mean))
    for i in range(len(y_test)):
        input_ = values[i:i+4]
        print(np.array([input_]))
        print(np.reshape(np.fliplr(np.array([input_])), (-1,)))

        # input_ = x_test[i, 0, :]
        price_sigma, price_mean_wts, price_cov_wts = price_ukf.calc_sigma_points(mean=input_, cov=price_ukf.covar)

        param_sigma, param_mean_wts, param_cov_wts = wt_ukf.calc_sigma_points(mean=wt_ukf.mean, cov=wt_ukf.cov)

        # Transition state variables
        new_price_sigma = price_ukf.state_transition(wt_ukf.mean, price_sigma.T, price_mean_wts, price_cov_wts)[0]

        new_param_sigma = wt_ukf.transition_state(param_sigma, param_mean_wts, param_cov_wts, np.mean(grad_data, axis=1), np.cov(grad_data))[0]

        # Calculating the output sigma points for the predicted price
        output_price_sigma, output_price_mean, output_price_cov = price_ukf.output(new_price_sigma, price_mean_wts, price_cov_wts, measurement_noise=np.mean(err_data), measure_covar=np.cov(err_data))
        # Should the input to the price ukf output be the market prices, because I'll always have the
        # last 4 market prices; I think this would be ideal, as that is what the nn is trained on

        # Calculating the output sigma points for the predicted gradient
        output_grad_sigma, output_grad_mean, output_grad_cov = wt_ukf.output_state(new_param_sigma, param_mean_wts, param_cov_wts, shapes, np.array([input_]), output_price_mean)
                                                                      # update_wt(self, input_prices, true_price, param_sigma, param_mean_wts, param_cov_weights, grad_sigma, grad_mean, grad_cov)
        # Getting the real value that is used to update both
        print(f'output price mean {output_price_mean} in iter {i}')
        measured_price = y_test[i, 0]
        print(f'true price mean {measured_price} in iter {i}')

        # Updating the price ukf
        price_ukf.update(measured_price, price_sigma, price_mean_wts, price_cov_wts, output_price_sigma, output_price_mean, output_price_cov)

        # Updating the weight ukf
        grad = wt_ukf.update_wt(np.array([input_]), measured_price, new_param_sigma, param_mean_wts, param_cov_wts, output_grad_sigma, output_grad_mean, output_grad_cov)
        wts = wt_ukf.model.get_weight_state()[0]

        '''THIS IS WHERE I STOPPED EDITING'''
        # Changing all the historical data used for covariances
        grad_data = np.append(grad_data, np.array([grad]).T, axis=1)[:, 1:]
        err_data = np.append(err_data, output_price_mean - measured_price)[1:]
        price_data = np.append(price_data, np.array([input_]).T, axis=1)[:, 1:]
        wt_data = np.append(wt_data, np.array([wts]).T, axis=1)[:, 1:]


        print(values[i+7])
        values[i+7] = price_ukf.mean[0]
        print(values[i+7])
        print(f'finished iter {i}')
        # During price ukf state transition, process noise and cov must be scaled by learning rate

    plt.clf()
    plt.plot(y_test, label='Market')
    plt.plot(values[7:], label='Prediction')
    plt.legend()
    plt.show()



# A dual UKF is based on there being a prediction on the weights as well, which my prediction
# is just taking the mean of the last 10 weights. my real is going to be actually taking the
# and applying the gradient based on the real value. the question is then how do we calculate a kalman
# gain to find the 'real' value. the kalman gain is based on the cross covariance of the grad predictions
# and the covariance of the outputs
# In that case, we hold the input data constant and then calculate sigma points of the model weights
# In this case, we hold the weights constand and calc sigma points of the input data, and then the
# grads are the different outputs that we calculate, and then we update based on the grad of the real
# and then update with the grad calculated from the real output and the real grad
# w



if __name__ == '__main__':
    main()
