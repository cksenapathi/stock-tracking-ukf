from test_main import gather_data, split_data, unzip, login
from ukf import UKF
from model import Model
from optimizer import Optimizer
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt


def calc_covars(model, opt, x, y, history):
    x = x[-history:, :]
    y = y[-history:]
    weights = np.zeros((len(model.get_weight_state()[0]), history))
    grads = np.zeros_like(weights)
    measure = np.zeros(history)
    for i in range(history):
        grad = opt.grad(model=model, inputs=np.array([x[i, :]]), targets=np.array([y[i]]))[1]
        opt.optimizer.apply_gradients((zip(grad, model.get_trainable_variables())))
        delta = unzip(grad)
        grads[:, i] = delta
        weights[:, i] = model.get_weight_state()[0]
        measure[i] = y[i] - model.predict(np.array([x[i, :]]))
    return weights, grads, measure


def main():
    td_session = login()
    lr = .0001
    history = 10
    # file_path = True
    file1 = 'C:\\Users\\chinm\\PycharmProjects\\StockTracking\\open_model'
    file2 = 'C:\\Users\\chinm\\PycharmProjects\\StockTracking\\high_model'
    file3 = 'C:\\Users\\chinm\\PycharmProjects\\StockTracking\\low_model'
    file4 = 'C:\\Users\\chinm\\PycharmProjects\\StockTracking\\close_model'
    x, y = gather_data(td_session, 'AAPL', k=4)
    x_train, y_train, x_test, y_test, x_val, y_val = split_data(x, y)
    x_train = np.transpose(x_train, (0, 2, 1))
    x_test = np.transpose(x_test, (0, 2, 1))
    x_val = np.transpose(x_val, (0, 2, 1))
    # Calculate all associated covariances big gay
    np.save('C:\\Users\\chinm\\PycharmProjects\\StockTracking\\x_test.npy', x_test)
    np.save('C:\\Users\\chinm\\PycharmProjects\\StockTracking\\y_test.npy', y_test)

    np.save('C:\\Users\\chinm\\PycharmProjects\\StockTracking\\x_train.npy', x_train)
    np.save('C:\\Users\\chinm\\PycharmProjects\\StockTracking\\y_train.npy', y_train)

    np.save('C:\\Users\\chinm\\PycharmProjects\\StockTracking\\y_val.npy', y_val)
    np.save('C:\\Users\\chinm\\PycharmProjects\\StockTracking\\x_val.npy', x_val)

    model1 = Model(filepath=file1)
    model2 = Model(filepath=file2)
    model3 = Model(filepath=file3)
    model4 = Model(filepath=file4)
    op_wts, shapes = model1.get_weight_state()
    hi_wts, _ = model2.get_weight_state()
    lo_wts, _ = model3.get_weight_state()
    cl_wts, _ = model4.get_weight_state()
    models = [model1, model2, model3, model4]
    paths = [file1, file2, file3, file4]
    grad_data = np.zeros((4, len(op_wts), history))
    weight_data = np.zeros((4, len(op_wts), history))
    err_data = np.zeros((4, history))
    opt = Optimizer()
    ukfs = []
    for i, model in enumerate(models):
        weights, grad, err = calc_covars(model, opt, x_test[:, i, :], y_test[:, i], history=history)
        grad_data[i, :, :] = grad
        weight_data[i, :, :] = weights
        err_data[i, :] = err
        ukfs.append(UKF(mean=np.mean(weights, axis=1), covariance=np.cov(weights), model=model, shapes=shapes))
    grad_data *= lr
    predictions = []
    for i, ukf in enumerate(ukfs):
        temp_model = Model(filepath=paths[0])
        sigma, mean_weights, covar_weights = ukf.calc_sigma_points(np.mean(weight_data[i, :, :], axis=1),
                                                                   np.cov(weight_data[i, :, :]))
        plt.figure(1)
        plt.plot(temp_model.predict(x_test[:, i, :]), label='pred high')
        temp_model.set_weights(shapes, model1.get_weight_state()[0])
        plt.plot(temp_model.predict(x_test[:, i, :]), label='reset weights')
        plt.plot(y_test[:, i], label='real')
        plt.legend()
        plt.show()
        pred = np.zeros_like(y_test[:, i])
        err_mean = np.mean(err_data[i, :])
        err_cov = np.cov(err_data[i, :])
        for j in range(len(x_test)):
            predict_output, predict_output_mean, predict_output_covar = ukf.output(x=np.array([x_test[j, :, i]]),
                                                                                   sigma=sigma.T,
                                                                                   mean_weights=mean_weights,
                                                                                   covar_weights=covar_weights,
                                                                                   measurement_noise=err_mean,
                                                                                   measure_covar=err_cov)
            print(np.mean(predict_output))
            print(models[i].predict(np.array([x_test[j, :, i]])))
            pred[j] = predict_output_mean
            ukf.update(y_test[j, i], sigma, mean_weights, covar_weights, predict_output, predict_output_mean,
                       predict_output_covar)
            ukf.output_mat.set_weights(shapes, ukf.mean)
            grad = lr * unzip(opt.grad(models[i], np.array([x_test[j, :, i]]), np.array([y_test[j, i]]))[1])
            grad_data[i, :, :] = np.append(grad_data[i, :, :], np.array([grad]).T, axis=1)[:, 1:]
            weight_data[i, :, :] = np.append(weight_data[i, :, :], np.array([models[i].get_weight_state()[0]]).T,
                                             axis=1)[:, 1:]
            err_data[i, :] = np.append(err_data[i, :], y_test[j, i] - predict_output_mean)[1:]
            # wt_mean = np.mean(weight_data[i, :, :], axis=1)
            wt_covar = np.cov(weight_data[i, :, :])

            grad_mean = np.mean(grad_data[i, :, :], axis=1)
            grad_covar = np.cov(grad_data[i, :, :])

            err_mean = np.mean(err_data[i, :])
            err_cov = np.cov(err_data[i, :])

            print(f'weight covar {wt_covar}\nerr mean {err_mean} err cov {err_cov} \ngrad {grad_mean} \ngrad cov \n{grad_covar}\n\n')
            print(f'mean before state transition {ukf.mean}')
            sigma, mean_weights, covar_weights = ukf.calc_sigma_points(models[i].get_weight_state()[0], wt_covar)
            sigma = ukf.state_transition(sigma=sigma, mean_weights=mean_weights, covar_weights=covar_weights,
                                         process_noise=grad_mean, process_covar=grad_covar)[0]
            print(f'mean after state transition {ukf.mean}')
            print()
            temp_model.set_weights(shapes=shapes, i=ukf.mean)
            plt.figure(1)
            plt.plot(temp_model.predict(x_test[:, i, :]), label=f'pred {j}')
            plt.legend()
            plt.show()
        predictions.append(pred)

    plt.cla()
    fig, ((p1, p2), (p3, p4)) = plt.subplots(2, 2)
    plt.suptitle('Best Weights from Monte Carlo Approximation: Test Set')
    p1.plot(y_test[:, 0], label='Real')
    p1.plot(predictions[0], label='Pred')
    p1.title.set_text('Open')
    p1.legend()

    p2.plot(y_test[:, 1], label='Real')
    p2.plot(predictions[1], label='Pred')
    p2.title.set_text('High')
    p2.legend()

    p3.plot(y_test[:, 2], label='Real')
    p3.plot(predictions[2], label='Pred')
    p3.title.set_text('Low')
    p3.legend()

    p4.plot(y_test[:, 3], label='Real')
    p4.plot(predictions[3], label='Pred')
    p4.title.set_text('Close')
    p4.legend()
    plt.show()


if __name__ == '__main__':
    main()
