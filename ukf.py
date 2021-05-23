import numpy as np
import tensorflow as tf
from model import Model
from optimizer import Optimizer
from utils import *

# Mean and covariance is given by the last 4 prices
# Transition is based on the nn trained, which must be updated after each iter
# Output is taking the 0th index, and adding the mean measurement error
# Update is taking the last 4 market prices and updating with kalman gain and covariances

class UKF:
    def __init__(self, mean, covariance, model: Model, shapes, alpha=.8, beta=2, kappa=.001):
        self.mean = mean
        self.covar = covariance
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.transition_mat = model
        self.shapes = shapes

    def get_state(self):
        return self.mean, self.covar

    def calc_sigma_points(self, mean, cov):
        # noinspection PyPep8Naming
        L = len(mean)
        lam = self.alpha ** 2 * (L + self.kappa) - L
        sigma = np.tile(mean, (2 * L + 1, 1))
        if cov is None:
            mat = np.zeros((L, L))
        else:
            try:
                mat = (L + lam) * cov
                mat = (mat + mat.T)/2
                v = np.linalg.cholesky(mat).T
                print(v)
            except np.linalg.LinAlgError:
                try:
                    mat = (L + lam) * cov
                    mat = (mat + mat.T)/2
                    v = np.linalg.cholesky(mat + 5 * np.mean(np.diag(mat)) * np.eye(L)).T
                    print(v)
                except np.linalg.LinAlgError:
                    print('price ukf cholesky failed')
                    print(f'cov {cov}')
                    print(f'mean: {np.mean(cov)} covar: {(L + lam) * cov}')
                    print(f'trace: {np.sum(np.diag((mat + mat.T)/2))}, mean trace: {np.mean(np.diag((mat + mat.T)/2))}')
                    print(f'L + lam {L + lam}')
                    print((mat +mat.T)/2)
                    exit(1)
        sigma[0] = mean
        sigma[1:L + 1, :] += v
        sigma[L + 1:, :] -= v
        mean_weights = np.zeros(2*L+1)
        covar_weights = np.zeros(2*L+1)
        # mean_weights[0], covar_weights[0] = 1, 1
        # for i in range(1, 2*L+1):
        #     mean_weights[i] = np.exp(-(((i % L)/L) ** 2))
        #     covar_weights[i] = np.exp(-(((i % L)/L) ** 2))

        mean_weights[0] = lam/(L + lam)
        covar_weights[0] = (lam/(L + lam)) + (1- self.alpha**2 + self.beta)
        for i in range(1, 2*L + 1):
            mean_weights[i] = 1/(2*(L * lam))
            covar_weights[i] = 1/(2*(L * lam))
        return sigma.T, mean_weights, covar_weights

    def set_state_transition_weights(self, params):
        self.transition_mat.set_weights(self.shapes, params)

    def state_transition(self, params, price_sigma, price_mean_wts, price_covar_wts):
        self.set_state_transition_weights(params=params)
        new_states  = np.zeros_like(price_sigma)
        for i, sig in enumerate(price_sigma):
            new_states[i, 1:] = sig[0:3]
            new_states[i, 0] = self.transition_mat.predict(np.array([sig]))
        new_mean = new_states.T @ price_mean_wts
        self.mean = new_mean
        var = new_states - np.tile(new_mean, (9,1))
        print(var.shape)
        self.covar = ((new_states - np.tile(new_mean, (9,1))).T * price_covar_wts) @ (new_states - np.tile(new_mean, (9,1)))
        return self.calc_sigma_points(mean=self.mean, cov=self.covar)

    def set_output_method(self, mat):
        if type(mat) is np.ndarray or type(mat) is tf.python.keras.engine.sequential.Sequential:
            self.output_mat = mat
        else:
            print('Check the data type of the state transition function')

    def output(self, sigma, mean_wts, covar_wts, measurement_noise=None, measure_covar=None):
        outputs = sigma[0, :]
        output_mean = np.dot(outputs, mean_wts) + measurement_noise
        output_covar = np.dot((outputs-output_mean) * covar_wts, (outputs-output_mean)) + measure_covar
        return outputs.T, output_mean, output_covar

    def update(self, measured_mean, state_sigma, state_mean_wts, state_cov_wts, output_sigma, output_mean, output_covar):
        state_mean = state_sigma @ state_mean_wts
        var1 = ((state_sigma - np.tile(state_mean, (max(state_sigma.shape), 1)).T) * state_cov_wts)
        var2 = (np.array([output_sigma]) - np.tile(np.array([output_mean]), (max(output_sigma.shape), 1)).T)
        print(f'left {var1.shape}, right {var2.shape}')
        cross_covar = ((state_sigma - np.tile(state_mean, (max(state_sigma.shape), 1)).T) * state_cov_wts) @ (np.array([output_sigma]) - np.tile(np.array([output_mean]), (max(output_sigma.shape), 1)).T).T
        kalman_gain = cross_covar @ np.linalg.inv(np.array([[output_covar]]))
        # print(measured_mean)
        # print(output_mean)
        # print((kalman_gain*(measured_mean - output_mean)).shape)
        # kalman_gain = np.reshape(kalman_gain, max(kalman_gain.shape))
        # print(self.mean.shape)
        self.mean +=  kalman_gain @ np.array([measured_mean - output_mean])
        self.covar -= kalman_gain @ np.array([[output_covar]]) @ kalman_gain.T
        return

# In order to make this work, I need to be able to change the weights so that the output model in this one and the
# transition model in the other one are the same weights. The other one is going to return the grad it believes it
# should apply and this one will return the weights it thinks it should have. The process noise should be the grad from
# the other one and the other one should take the weights of this one after
