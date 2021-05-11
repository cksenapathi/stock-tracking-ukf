import numpy as np
from model import Model
from optimizer import Optimizer


class WeightUKF:
    def __init__(self, mean, cov, model: Model, opt: Optimizer, alpha=.01, beta=2, kappa=0):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.mean = mean
        self.cov = cov
        self.model = model
        self.opt = opt

    def calc_sigma_points(self, mean, cov):
        # noinspection PyPep8Naming
        L = len(mean)
        lam = self.alpha ** 2 * (L + self.kappa) - L
        sigma = np.tile(mean, (2 * L + 1, 1))
        mat = np.array([])
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

    def transition_state(self, param_sigma, param_mean_wts, param_cov_wts, process_noise, process_cov):
        opt.optimizer.apply_gradients(zip(zipup(self.shapes, process_noise), self.model.get_trainable_variables()))
        self.mean = self.model.get_weight_state()[0]
        self.cov = ((param_sigma - np.tile(self.mean, (np.max(new_sigma.shape), 1)).T) * cov_weights) @ \
                   (param_sigma - np.tile(self.mean, (np.max(new_sigma.shape), 1)).T).T + (opt.lr ** 2) *process_cov
        return self.calc_sigma_points(mean=self.mean, cov=self.cov)

    def output_state(self, param_sigma, param_mean_wts, param_cov_wts, model_shapes, input_prices, target_price):
        param_sigma = param_sigma.T
        grad_sigma = np.zeros_like(param_sigma)
            self.model.set_weights(model_shapes, i)
            grad_sigma[i, :] = self.opt.grad(self.model, input_prices, target_price)
        grad_mean = grad_sigma.T @ param_mean_weights
        grad_cov = grad_sigma.T - np.tile(grad_mean, (1, max(grad_sigma.shape))) * cov_weights @ \
            grad_sigma - np.tile(grad_mean.T, (max(grad_sigma.shape), 1))
        self.model.set_weights(model_shapes, self.mean)
        return grad_sigma, grad_mean, grad_cov

    def update_wt(self, input_prices, true_price, param_sigma, param_mean_wts, param_cov_weights, grad_sigma, grad_mean, grad_cov):
        param_mean = param_sigma @ param_mean_wts
        true_grad = self.opt.grad(self.model, input_prices, true_price)
        cross_cov = ((param_sigma - np.tile(param_mean, (max(param_sigma.shape), 1))) * param_cov_weights) @\
                    (grad_sigma - np.tile(grad_mean, (max(grad_sigma.shape), 1)))
        kalman_gain = cross_cov @ np.linalg.inv(grad_cov)
        self.mean += kalman_gain @ (true_grad - grad_mean)
        self.cov -= kalman_gain @ grad_cov @ kalman_gain.T
        return self.mean
