import numpy as np
from model import Model
from optimizer import Optimizer
from utils import zipup, unzip


class WeightUKF:
    def __init__(self, mean, cov, model: Model, shapes, opt: Optimizer, alpha=.01, beta=2, kappa=0):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.mean = mean
        self.old_mean = mean
        self.cov = cov
        self.model = model
        self.shapes = shapes
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
                v = np.linalg.cholesky(mat)
            except np.linalg.LinAlgError:
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
        sigma[1:L + 1, :] = sigma[1:L + 1, :] + v
        sigma[L + 1:, :] = sigma[L + 1:, :] - v
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

    def transition_state(self, param_sigma, param_mean_wts, param_cov_wts, process_noise, process_cov):
        self.opt.optimizer.apply_gradients(zip(zipup(self.shapes, process_noise), self.model.get_trainable_variables()))
        self.mean = self.model.get_weight_state()[0]
        self.cov = ((param_sigma - np.tile(self.mean, (np.max(param_sigma.shape), 1)).T) * param_cov_wts) @ \
                   (param_sigma - np.tile(self.mean, (np.max(param_sigma.shape), 1)).T).T + (self.opt.lr ** 2) * process_cov
        self.cov = .5 * (self.cov + self.cov.T)
        return self.calc_sigma_points(mean=self.mean, cov=self.cov)

    def output_state(self, param_sigma, param_mean_wts, param_cov_wts, model_shapes, input_prices, target_price):
        param_sigma = param_sigma.T
        grad_sigma = np.zeros_like(param_sigma)
        for i, sigma in enumerate(param_sigma):
            self.model.set_weights(model_shapes, sigma)
            grad_sigma[i, :] = unzip(self.opt.grad(self.model, input_prices, target_price)[1])
        grad_mean = grad_sigma.T @ param_mean_wts
        var = grad_sigma - np.tile(grad_mean, (max(grad_sigma.shape), 1))
        print(var.shape)
        grad_cov = ((grad_sigma - np.tile(grad_mean, (max(grad_sigma.shape), 1))).T * param_cov_wts) @ \
            (grad_sigma - np.tile(grad_mean, (max(grad_sigma.shape), 1)))
        # grad_cov = np.cov(grad_sigma.T)
        print(f'in output, {grad_cov.shape}')
        self.model.set_weights(model_shapes, self.mean)
        return grad_sigma, grad_mean, grad_cov

    def update_wt(self, input_prices, true_price, param_sigma, param_mean_wts, param_cov_weights, grad_sigma, grad_mean, grad_cov):
        param_mean = param_sigma @ param_mean_wts
        true_grad = unzip(self.opt.grad(self.model, input_prices, true_price)[1])
        cross_cov = ((param_sigma - np.tile(param_mean.T, (max(param_sigma.shape), 1)).T) * param_cov_weights) @\
                    (grad_sigma - np.tile(grad_mean, (max(grad_sigma.shape), 1)))
        print('grad covar det: {}'.format(np.linalg.det(grad_cov)))
        print(grad_cov.shape)
        try:
            kalman_gain = cross_cov @ np.linalg.inv(grad_cov)
        except np.linalg.LinAlgError:
            _, val = np.linalg.eigh(grad_cov)
            grad_cov = grad_cov - np.tile(val[:, -1], (len(grad_cov), 1))
            print(np.linalg.slogdet(grad_cov))
            print(np.linalg.eigh(grad_cov)[0])
            # while np.isclose(np.linalg.det(grad_cov), 0):
            #     _, val = np.linalg.eigh(grad_cov)
            #     grad_cov = grad_cov - np.tile(val[:, -1], (len(grad_cov), 1))
            #     print(np.linalg.det(grad_cov))
            kalman_gain = cross_cov @ np.linalg.inv(grad_cov)
        self.mean += kalman_gain @ (true_grad - grad_mean)
        self.cov -= kalman_gain @ grad_cov @ kalman_gain.T
        grad = (self.mean - self.old_mean)/self.opt.lr
        print('grad in update {}'.format(grad))
        self.old_mean = self.mean
        return grad
