from ukf import UKF
import numpy as np
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import tensorflow.python.ops.numpy_ops.np_config as np_config
import time
from td.client import TDClient
from optimizer import Optimizer
from wt_ukf import WeightUKF
from utils import gather_data, login, split_data, unzip


def main2():
    from sklearn.ensemble import RandomForestRegressor

    td_session = TDClient(
        client_id='5WINVND6ZU0XRIELK5DRJLHZGK9KGYGB',
        redirect_uri='http://localhost',
        credentials_path='C:\\Users\\chinm\\PycharmProjects\\StockTracking\\td_cred.json'
    )
    # best_errors = np.array([np.inf, np.inf, np.inf])
    # best_weights = [1] * 3
    # iters = 100  # Number of Monte Carlo Iterations
    x, y = gather_data(td_session, 'AAPL', k=4)
    x_train, y_train, x_test, y_test, x_val, y_val = split_data(x, y)
    x_train = np.reshape(x_train, newshape=(-1, 16))
    x_test = np.reshape(x_test, newshape=(-1, 16))
    x_val = np.reshape(x_val, newshape=(-1, 16))
    # for _ in range(3):
    forest = RandomForestRegressor(n_estimators=200)
    forest.fit(x_train, y_train)
    print(x_val)
    print(forest.predict(np.array([x_val[0]])))
    print(forest.score(x_test, y_test))
    plt.figure(1)
    plt.plot(y_val, label='real')
    plt.plot(forest.predict(x_val), label='pred')
    plt.legend()
    plt.show()


def update_best(out, best_errors, best_weights, model):
    if out < best_errors[0]:
        best_errors[0] = out
        best_weights[0] = model.get_weights()
    elif out < best_errors[1]:
        best_errors[1] = out
        best_weights[1] = model.get_weights()
    elif out < best_errors[2]:
        best_errors[2] = out
        best_weights[2] = model.get_weights()


def main():
    td_session = TDClient(
        client_id='5WINVND6ZU0XRIELK5DRJLHZGK9KGYGB',
        redirect_uri='http://localhost',
        credentials_path='C:\\Users\\chinm\\PycharmProjects\\StockTracking\\td_cred.json'
    )
    file_path = True
    save = False
    file1 = 'C:\\Users\\chinm\\PycharmProjects\\StockTracking\\open_model'
    file2 = 'C:\\Users\\chinm\\PycharmProjects\\StockTracking\\high_model'
    file3 = 'C:\\Users\\chinm\\PycharmProjects\\StockTracking\\low_model'
    file4 = 'C:\\Users\\chinm\\PycharmProjects\\StockTracking\\close_model'
    x, y = gather_data(td_session, 'AAPL', k=4)
    x_train, y_train, x_test, y_test, x_val, y_val = split_data(x, y)
    x_train = np.transpose(x_train, (0, 2, 1))
    x_test = np.transpose(x_test, (0, 2, 1))
    x_val = np.transpose(x_val, (0, 2, 1))
    if not file_path:
        best_errors1 = np.array([np.inf, np.inf, np.inf])
        best_weights1 = [1] * 3

        best_errors2 = np.array([np.inf, np.inf, np.inf])
        best_weights2 = [1] * 3

        best_errors3 = np.array([np.inf, np.inf, np.inf])
        best_weights3 = [1] * 3

        best_errors4 = np.array([np.inf, np.inf, np.inf])
        best_weights4 = [1] * 3
        iters = 10  # Number of Monte Carlo Iterations
        for i in range(iters):
            model1 = Sequential()
            model1.add(Dense(4))
            model1.add(Dense(8))
            model1.add(Dense(1))
            model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.00001),
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=['mean_squared_error'])
            model1.fit(x_train[:, 0, :], y_train[:, 0], batch_size=1, epochs=3)
            out = model1.evaluate(x_test[:, 0, :], y_test[:, 0], batch_size=1)[0]
            update_best(out, best_errors1, best_weights1, model1)

            model2 = Sequential()
            model2.add(Dense(4))
            model2.add(Dense(8))
            model2.add(Dense(1))
            model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.00001),
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=['mean_squared_error'])
            model2.fit(x_train[:, 1, :], y_train[:, 1], batch_size=1, epochs=3)
            out = model2.evaluate(x_test[:, 1, :], y_test[:, 1], batch_size=1)[0]
            update_best(out, best_errors2, best_weights2, model2)

            model3 = Sequential()
            model3.add(Dense(4))
            model3.add(Dense(8))
            model3.add(Dense(1))
            model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.00001),
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=['mean_squared_error'])
            model3.fit(x_train[:, 2, :], y_train[:, 2], batch_size=1, epochs=3)
            out = model3.evaluate(x_test[:, 3, :], y_test[:, 3], batch_size=1)[0]
            update_best(out, best_errors3, best_weights3, model3)

            model4 = Sequential()
            model4.add(Dense(4))
            model4.add(Dense(8))
            model4.add(Dense(1))
            model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.00001),
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=['mean_squared_error'])
            model4.fit(x_train[:, 3, :], y_train[:, 3], batch_size=1, epochs=3)
            out = model4.evaluate(x_test[:, 3, :], y_test[:, 3], batch_size=1)[0]
            update_best(out, best_errors4, best_weights4, model4)

        print(best_errors1)
        print(best_errors2)
        print(best_errors3)
        print(best_errors4)
        # for i in range(len(best_errors)):
        #     if not best_errors[i] == np.inf:
        #         model.set_weights(best_weights[i])
        #         model.save(filepath=f'C:\\Users\\chinm\\PycharmProjects\\StockTracking\\model{i+1}')
        model1.set_weights(best_weights1[0])
        model2.set_weights(best_weights2[0])
        model3.set_weights(best_weights3[0])
        model4.set_weights(best_weights4[0])
        if save:
            model1.save(filepath='C:\\Users\\chinm\\PycharmProjects\\StockTracking\\open_model')
            model2.save(filepath='C:\\Users\\chinm\\PycharmProjects\\StockTracking\\high_model')
            model3.save(filepath='C:\\Users\\chinm\\PycharmProjects\\StockTracking\\low_model')
            model4.save(filepath='C:\\Users\\chinm\\PycharmProjects\\StockTracking\\close_model')
    else:
        model1 = tf.keras.models.load_model(file1)
        model2 = tf.keras.models.load_model(file2)
        model3 = tf.keras.models.load_model(file3)
        model4 = tf.keras.models.load_model(file4)

    model1.summary()
    plt.cla()
    fig, ((p1, p2), (p3, p4)) = plt.subplots(2, 2)
    plt.suptitle('Best Weights from Monte Carlo Approximation: Test Set')
    p1.plot(y_test[:, 0], label='Real')
    p1.plot(model1.predict(x_test[:, 0, :]), label='Pred')
    p1.title.set_text('Open')
    p1.legend()

    p2.plot(y_test[:, 1], label='Real')
    p2.plot(model2.predict(x_test[:, 1, :]), label='Pred')
    p2.title.set_text('High')
    p2.legend()

    p3.plot(y_test[:, 2], label='Real')
    p3.plot(model3.predict(x_test[:, 2, :]), label='Pred')
    p3.title.set_text('Low')
    p3.legend()

    p4.plot(y_test[:, 3], label='Real')
    p4.plot(model4.predict(x_test[:, 3, :]), label='Pred')
    p4.title.set_text('Close')
    p4.legend()
    plt.show()

    plt.cla()
    fig, ((p1, p2), (p3, p4)) = plt.subplots(2, 2)
    plt.suptitle('Best Weights from Monte Carlo Approximation: Validation Set')
    p1.plot(y_val[:, 0], label='Real')
    p1.plot(model1.predict(x_val[:, 0, :]), label='Pred')
    p1.title.set_text('Open')
    p1.legend()

    p2.plot(y_val[:, 1], label='Real')
    p2.plot(model2.predict(x_val[:, 1, :]), label='Pred')
    p2.title.set_text('High')
    p2.legend()

    p3.plot(y_val[:, 2], label='Real')
    p3.plot(model3.predict(x_val[:, 2, :]), label='Pred')
    p3.title.set_text('Low')
    p3.legend()

    p4.plot(y_val[:, 3], label='Real')
    p4.plot(model4.predict(x_val[:, 3, :]), label='Pred')
    p4.title.set_text('Close')
    p4.legend()
    plt.show()

    # plt.plot(y_test, label='Real')
    # plt.plot(model.predict(x_test, training=False)[:, 0, :], label='Pred')
    # plt.legend()
    # plt.show()


def main1():
    print('starting main')
    history = 10
    td_session = TDClient(
        client_id='5WINVND6ZU0XRIELK5DRJLHZGK9KGYGB',
        redirect_uri='http://localhost',
        credentials_path='C:\\Users\\chinm\\PycharmProjects\\StockTracking\\td_cred.json'
    )
    x, y = gather_data(td_session=td_session, tic='AAPL', k=4)
    x_train, y_train, x_test, y_test, x_val, y_val = split_data(x, y)
    plt.plot(y_train)
    # file_path = 'C:\\Users\\chinm\\PycharmProjects\\StockTracking\\model1'
    file_path = None
    model = Model(filepath=file_path)
    opt = Optimizer()
    print('got here')
    epochs = 2
    for epoch in range(epochs):
        for i in range(len(x_train)):
            _, grad = opt.grad(model, np.array([x_train[i, :, :]]), np.array([y_train[i, :]]))
            if file_path is None:
                opt.optimizer.apply_gradients((zip(grad, model.get_trainable_variables())))
    print('finished training')

    delta = unzip(grad)
    old_weights = np.zeros((len(delta), history))
    old_errors = np.zeros((4, history))
    old_grads = np.zeros((len(delta), history))
    np_config.enable_numpy_behavior()
    for i in range(len(x_train)):
        _, grad = opt.grad(model, np.array([x_train[i, :, :]]), np.array([y_train[i, :]]))
        opt.optimizer.apply_gradients(zip(grad, model.get_trainable_variables()))
        delta = unzip(grad)

        # print(f'old grads: {old_grads.shape} grad: {np.array([delta]).T.shape}')
        old_grads = np.append(old_grads, np.array([delta]).T, axis=1)[:, 1:]
        old_weights = np.append(old_weights, np.array([unzip(model.get_trainable_variables())]).T, axis=1)[:, 1:]
        old_errors = np.append(old_errors, ((y_train[i, :] - model.predict(np.array([x_train[i, :, :]]),
                                                                           training=False))[0]).T, axis=1)[:, 1:]
    print('finished making old value matrices')
    print('calculating mean and covariances')
    wt_mean = np.mean(old_weights, axis=1)
    wt_covar = (old_weights - np.tile(wt_mean, (history, 1)).T) @ (old_weights - np.tile(wt_mean, (history, 1)).T).T
    wt_covar /= history

    proc_mean = np.mean(old_grads, axis=1)
    proc_covar = (old_grads - np.tile(proc_mean, (history, 1)).T) @ (old_grads - np.tile(proc_mean, (history, 1)).T).T
    proc_covar /= history

    m_mean = np.mean(old_errors, axis=1)
    m_covar = (old_errors - np.tile(m_mean, (history, 1)).T) @ (old_errors - np.tile(m_mean, (history, 1)).T).T
    m_covar /= history

    print('initializing ukf')
    state, shape = model.get_weight_state()
    ukf = UKF(mean=wt_mean, covariance=wt_covar, model=model, shapes=shape)
    sigma, mean_weights, covar_weights = ukf.calc_sigma_points(wt_mean, wt_covar)
    predicted_y = np.zeros((1, 4))
    print('ukf started, starting loop')
    start = time.time()
    # ani = FuncAnimation(fig, run, fargs=(predicted_y, y_test), interval=16000, blit=True)
    # plt.show()
    len(x_test)
    for i in range(len(x_test)):
        # Predicts output based on current weight state
        predict_output, predict_output_mean, predict_output_covar = ukf.output(x=np.array([x_test[i, :, :]]),
                                                                               sigma=sigma,
                                                                               mean_weights=mean_weights,
                                                                               covar_weights=covar_weights,
                                                                               measurement_noise=m_mean,
                                                                               measure_covar=m_covar)
        # Get and insert real values
        print(predict_output_mean)
        predicted_y = np.append(predicted_y, np.array([predict_output_mean]), axis=0)
        print('output predicted')
        ukf.update(y_test[i, :], sigma, mean_weights, covar_weights, predict_output, predict_output_mean,
                   predict_output_covar)
        print('real value updated')
        ukf.output_mat.set_weights(shape, ukf.mean)
        time_past = time.time() - start
        start = time.time()
        # Update all noise means and covars
        _, grad = opt.grad(model, np.array([x_test[i, :, :]]), np.array([y_test[i, :]]))
        grad = unzip(grad)
        print(f'old grad {old_grads.shape} grad {grad.shape}')
        old_grads = np.append(old_grads, np.array([grad]).T, axis=1)[:, 1:]
        old_weights = np.append(old_weights, np.array([model.get_weight_state()[0]]).T, axis=1)[:, 1:]
        old_errors = np.append(old_errors, np.array([y_train[i, :] - predict_output_mean.T]).T, axis=1)[:, 1:]
        print('old values updated')
        wt_covar = (old_weights - np.tile(np.mean(old_weights, axis=1), (history, 1)).T) @ (
                old_weights - np.tile(np.mean(old_weights, axis=1), (history, 1)).T).T
        wt_covar /= history
        proc_covar = (old_grads - np.tile(np.mean(old_grads, axis=1), (history, 1)).T) @ (
                old_grads - np.tile(np.mean(old_grads, axis=1), (history, 1)).T).T
        proc_covar /= history
        m_mean = np.mean(old_errors, axis=1)
        m_covar = (old_errors - np.tile(np.mean(old_errors, axis=1), (history, 1)).T) @ (
                old_errors - np.tile(np.mean(old_errors, axis=1), (history, 1)).T).T
        m_covar /= history
        print('covars recalculated')
        # This is where the state transition begins
        mean, _ = ukf.get_state()
        sigma, mean_weights, covar_weights = ukf.calc_sigma_points(mean, wt_covar)
        sigma = ukf.state_transition(sigma=sigma, mean_weights=mean_weights, covar_weights=covar_weights,
                                     process_noise=grad, process_covar=proc_covar)[0]

        print('state updated')
        print('finished iteration {} of {} after {} seconds'.format(i + 1, len(x_test), time_past))
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

    # todo figure out this real time graphing thing, because nothing else has worked so far
    fig, ((p1, p2), (p3, p4)) = plt.subplots(2, 2)
    print(predicted_y.shape)
    p1.plot(y_test[1:len(predicted_y), 0], label='Real')
    p1.plot(predicted_y[1:, 0], label='Pred')
    p1.set_ylim(100, 150)
    p1.legend()

    p2.plot(y_test[1:len(predicted_y), 1], label='Real')
    p2.plot(predicted_y[1:, 1], label='Pred')
    p2.set_ylim(100, 150)
    p2.legend()

    p3.plot(y_test[1:len(predicted_y), 2], label='Real')
    p3.plot(predicted_y[1:, 2], label='Pred')
    p3.set_ylim(100, 150)
    p3.legend()

    p4.plot(y_test[1:len(predicted_y), 3], label='Real')
    p4.plot(predicted_y[1:, 3], label='Pred')
    p4.set_ylim(100, 150)
    p4.legend()

    plt.show()
    print('finished testing loop')


if __name__ == '__main__':
    main()  # NN testing
    # main1()  # UKF testing
    # main2()  # Random Forest Regression testing
