import numpy as np
import tensorflow as tf
import pandas as pd
from ukf import UKF
from model import Model
from optimizer import Optimizer
from utils import *
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
    mod = Sequential()
    mod.add(Dense(4))
    mod.add(Dense(8))
    mod.add(Dense(1))
    print(mod(np.array([x_test[0, 0, :]])))
    lr = .001
    model1 = Model(model=mod)

    # history = 5
    # param_hist = 69
    # wt_data, grad_data, err_data = calc_covars(model1, opt, x_test[:, 0, :], y_test[:, 0], history, param_hist)
    # price_data = x_train[-history:, 0, :].T
    # price_mean = price_data[:, -1]
    # price_cov = np.cov(price_data)
    values = np.zeros(len(y_test)+7)
    wts, shapes = model1.get_weight_state()
    values[0:4] = x_test[0, 0, :]
    values[4:7] = x_test[1:4,0,0]
    print(values[:8])
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

    input_ = lrflip(values[0:4])
    wt_mean, wt_cov = wt_update.update(wts)
    price_mean, price_cov = price_update.update(input_)

    price_ukf = UKF(mean=lrflip(values[0:4]), covariance=price_cov, model=model1, shapes=shapes)
    wt_ukf = WeightUKF(mean=wt_mean, cov=wt_cov, model=model1, shapes=shapes, opt=Optimizer(lr=lr))
    # What do I use to update
    # grad_mean = np.zeros((len(wt_mean), 1))
    # grad_cov = np.zeros((len(wt_mean), len(wt_mean))) + .1 * np.eye(len(wt_mean))
    for i in range(len(y_test)):

        # input_ = x_test[i, 0, :]
        price_sigma, price_mean_wts, price_cov_wts = price_ukf.calc_sigma_points(mean=input_, cov=price_cov)

        param_sigma, param_mean_wts, param_cov_wts = wt_ukf.calc_sigma_points(mean=wt_update.mean, cov=wt_update.cov)

        # Transition state variables
        new_price_sigma = price_ukf.state_transition(wt_ukf.mean, price_sigma.T, price_mean_wts, price_cov_wts)[0]

        new_param_sigma = wt_ukf.transition_state(param_sigma, param_mean_wts, param_cov_wts, grad_update.mean, grad_update.cov)[0]

        # Calculating the output sigma points for the predicted price
        output_price_sigma, output_price_mean, output_price_cov = price_ukf.output(new_price_sigma, price_mean_wts, price_cov_wts, measurement_noise=err_update.mean, measure_covar=err_update.cov)

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
        grad_update.update(grad)
        wt_update.update(wt_ukf.mean)

        print(values[i+7])
        values[i+7] = price_ukf.mean[0]
        print(values[i+7])

        err_update.update(price_ukf.mean[0] - measured_price)
        print(f'finished iter {i}')
        input_ = lrflip(values[i+1:i+5])
        price_update.update(input_)
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
