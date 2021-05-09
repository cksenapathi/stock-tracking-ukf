import numpy as np
import tensorflow as tf
from model import Model
from optimizer import Optimizer
from utils import *


class UKF:
    def __init__(self, mean, covariance, model: Model, shapes, alpha=.8, beta=2, kappa=.001):
        self.mean = mean
        self.covar = covariance
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.transition_mat = np.identity(len(mean))
        self.output_mat = model
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
                v = np.linalg.cholesky(mat + 5 * np.mean(np.diag(mat)) * np.eye(L))
            except np.linalg.LinAlgError:
                print('cholesky failed')
                print(f'mean: {np.mean(cov)} covar: {np.cov(cov)}, \n{cov}')
                print(f'trace: {np.sum(np.diag((mat + mat.T)/2))}, mean trace: {np.mean(np.diag((mat + mat.T)/2))}')
                print(f'L + lam {L + lam}')
                print(mat - mat.T)
                exit(1)
        sigma[0] = mean
        sigma[1:L + 1, :] += v
        sigma[L + 1:, :] -= v
        mean_weights = np.zeros(2*L+1)
        covar_weights = np.zeros(2*L+1)
        mean_weights[0], covar_weights[0] = 1, 1
        for i in range(1, 2*L+1):
            mean_weights[i] = np.exp(-(((i % L)/L) ** 2))
            covar_weights[i] = np.exp(-(((i % L)/L) ** 2))

        # mean_weights = np.tile(np.array([1/(2*L+1)]), 2 * L + 1)
        # covar_weights = np.tile(np.array([1/(2*L+1)]), 2 * L + 1)
        return sigma.T, mean_weights/np.sum(mean_weights), covar_weights/np.sum(covar_weights)

    def set_state_transition(self, mat):
        if type(mat) is np.ndarray or type(mat) is tf.python.keras.engine.sequential.Sequential:
            self.transition_mat = mat
        else:
            print('Check the data type of the state transition function')

    def state_transition(self, sigma, mean_weights, covar_weights, process_noise, process_covar, opt: Optimizer):
        opt.optimizer.apply_gradients(zip(zipup(self.shapes, process_noise), self.output_mat.get_trainable_variables()))
        # delta = unzip(self.output_mat.get_trainable_variables()) - self.mean
        new_mean = unzip(self.output_mat.get_trainable_variables())
        new_covar = ((sigma - np.tile(new_mean, (np.max(sigma.shape), 1)).T) * covar_weights) @ \
                    (sigma - np.tile(new_mean, (np.max(sigma.shape), 1)).T).T + process_covar
        new_covar /= (2 * len(sigma[0]) + 1)
        self.mean = new_mean
        self.covar = new_covar
        self.output_mat.set_weights(shapes=self.shapes, i=self.mean)

        return self.calc_sigma_points(mean=new_mean, cov=new_covar)

    def set_output_method(self, mat):
        if type(mat) is np.ndarray or type(mat) is tf.python.keras.engine.sequential.Sequential:
            self.output_mat = mat
        else:
            print('Check the data type of the state transition function')

    def output(self, x, sigma, mean_weights, covar_weights, measurement_noise=None, measure_covar=None):
        size = np.max(sigma.shape)
        print(f'size {size}')
        outputs = np.zeros(size)
        print(f'sigma shape in output {sigma.shape}')
        for i, sig in enumerate(sigma):
            self.output_mat.set_weights(shapes=self.shapes, i=sig)
            outputs[i] = self.output_mat.predict(x=x, training=False)
            print(f'output {outputs[i]}')
            # print(outputs[:, i])
        print(f'outputs: {outputs.shape} mean weights: {mean_weights.shape} m_noise: {measurement_noise.shape}')
        output_mean = outputs @ mean_weights + measurement_noise
        print(f'outputs: {(outputs - np.tile(output_mean, (size, 1)).T).shape} mean weights: {covar_weights.shape} m_covar: {measure_covar.shape}')
        output_covar = (outputs - np.tile(output_mean, (size, 1)).T) * covar_weights
        print(output_covar.shape)
        output_covar = output_covar @ (outputs - np.tile(output_mean, (size, 1)).T).T + measure_covar
        print(f'output covar {output_covar}')
        return outputs, output_mean, output_covar

    def update(self, measured_mean, param_sigma, mean_wts, covar_weights, output_sigma, output_mean, output_covar):
        print(f'in update, output covar {output_covar} shape {output_covar.shape}')
        param_mean = param_sigma @ mean_wts
        cross_covar = ((param_sigma - np.tile(param_mean, (len(param_sigma), 1))) * covar_weights) @ \
                      (output_sigma - np.tile(output_mean, (len(output_sigma.T), 1)).T).T
        kalman_gain = cross_covar @ np.linalg.inv(output_covar)
        print(measured_mean)
        print(output_mean)
        print((kalman_gain*(measured_mean - output_mean)).shape)
        kalman_gain = np.reshape(kalman_gain, max(kalman_gain.shape))
        print(self.mean.shape)
        self.mean = self.mean + kalman_gain * (measured_mean - output_mean)
        self.covar -= kalman_gain * output_covar * kalman_gain.T
        print(self.mean)
        self.output_mat.set_weights(shapes=self.shapes, i=self.mean)
        return self.mean, self.shapes

# In order to make this work, I need to be able to change the weights so that the output model in this one and the
# transition model in the other one are the same weights. The other one is going to return the grad it believes it
# should apply and this one will return the weights it thinks it should have. The process noise should be the grad from
# the other one and the other one should take the weights of this one after
