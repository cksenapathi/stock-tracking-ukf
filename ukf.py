import numpy as np
import tensorflow as tf
from model import Model
'''FUCK TRYNA MAKE THIS GENERAL WRITE IT FOR YOUR PURPOSE'''


class UKF:
    def __init__(self, mean, covariance, model: Model, alpha=.1, beta=2, kappa=.001):
        self.mean = mean
        self.covar = covariance
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.transition_mat = np.identity(len(mean))
        self.output_mat = model

    def get_state(self):
        return self.mean, self.covar

    def calc_sigma_points(self, mean, covar):
        # noinspection PyPep8Naming
        L = len(mean)
        lam = self.alpha ** 2 * (L + self.kappa) - L
        sigma = np.tile(mean, (2 * L + 1, 1))
        mat = np.linalg.cholesky((L + lam) * np.abs(covar))
a         sigma[0] = mean
        sigma[1:L + 1, :] += mat
        sigma[L + 1:, :] -= mat
        mean_weights = np.append(np.array([lam / (L + lam)]),
                                 np.tile(1 / (2 * (L + lam)), 2 * L))
        covar_weights = np.append(np.array([lam / (L + lam) + (1 - self.alpha ** 2 + self.beta)]),
                                  np.tile(1 / (2 * (L + lam)), 2 * L))
        return sigma.T, mean_weights, covar_weights

    def set_state_transition(self, mat):
        if type(mat) is np.ndarray or type(mat) is tf.python.keras.engine.sequential.Sequential:
            self.transition_mat = mat
        else:
            print('Check the data type of the state transition function')

    def state_transition(self, sigma, mean_weights, covar_weights, process_noise=None):
        if process_noise is None:
            process_noise = np.zeros(len(self.mean))
        # Performs the potentially nonlinear state transition
        # All sigma points are taken through the state transition and the new
        # mean and covariance is computed based on the weights
        # if type(self.transition_mat) is np.ndarray:
        #     try:
        new_sigma = self.transition_mat @ sigma
        #     except ValueError:
        #         print("The dimensions of sigma and transition matrix don't match")
        # elif type(self.transition_mat) is tf.python.keras.engine.sequential.Sequential:
        #     try:
        #         new_sigma = self.transition_mat(sigma.T)
        #     except Exception as e:
        #         print(f'Tensorflow sucks: {e}')
        # else:
        #     print('Check the dimensions of your input the state transition')
        #     raise TypeError("The type of the transition rule doesn't work")
        new_mean = new_sigma @ mean_weights + process_noise
        new_covar = ((new_sigma - new_mean) @ covar_weights) @ (new_sigma - new_mean).T + np.outer(process_noise,
                                                                                                   process_noise)
        new_covar /= (2 * len(sigma[0]) + 1)
        return self.calc_sigma_points(mean=new_mean, covar=new_covar)
        # self.predict_mean = new_mean
        # self.predict_covar = new_covar

    def set_output_method(self, mat):
        if type(mat) is np.ndarray or type(mat) is tf.python.keras.engine.sequential.Sequential:
            self.output_mat = mat
        else:
            print('Check the data type of the state transition function')

    def output(self, shapes, x, sigma, mean_weights, covar_weights, measurement_noise=None):
        outputs = np.zeros((4, len(sigma)))
        if measurement_noise is None:
            measurement_noise = np.zeros(4)
        for i in sigma:
            self.output_mat.set_weights(weights=i)
            outputs[i] = self.output_mat.predict(x=x, training=False)
        output_mean = outputs @ mean_weights + measurement_noise
        output_covar = ((outputs - output_mean) @ covar_weights) @ (outputs - output_mean).T + \
            np.outer(np.abs(measurement_noise), np.abs(measurement_noise))
        return outputs, output_mean, output_covar

    def update(self, measured_mean, weights_sigma, weight_mean, covar_weights, output_sigma, output_mean, output_covar):
        cross_covar = ((weights_sigma - weight_mean) @ covar_weights) @ (output_sigma - output_mean).T
        kalman_gain = cross_covar @ np.linalg.inv(output_covar)
        self.mean += kalman_gain @ (measured_mean - output_mean)
        self.covar -= kalman_gain @ output_covar @ kalman_gain.T


# state transition is just identity matrix, process noise is added from the gradient vector
