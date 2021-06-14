# from td.client import TDClient
import pandas as pd
import math
import numpy as np


def prod(iter):
    val = 1
    for i in iter:
        val *= i
    return val


def update_mean(old_mean, new_data, count):
    return old_mean + new_data / (count + 1)


def update_cov(old_cov, new_mean, new_data, count):
    return (count / (count + 1)) * old_cov + (count/((count + 1) ** 2)) * np.outer(new_mean - new_data, new_mean - new_data)


def lrflip(vec):
    return np.reshape(np.fliplr(np.array([vec])), (-1,))


def calc_covars(model, opt, x, y, history, param_hist):
    x = x[-param_hist:, :]
    y = y[-param_hist:]
    history = param_hist - history
    weights = np.zeros((len(model.get_weight_state()[0]), param_hist))
    grads = np.zeros_like(weights)
    measure = np.zeros(param_hist - history)
    for i in range(param_hist):
        grad = opt.grad(model=model, inputs=np.array([x[i, :]]), targets=np.array([y[i]]))[1]
        opt.optimizer.apply_gradients((zip(grad, model.get_trainable_variables())))
        delta = unzip(grad)
        grads[:, i] = delta
        weights[:, i] = model.get_weight_state()[0]
        if i >= history:
            measure[i - history] = y[i] - model.predict(np.array([x[i, :]]))
    return weights, grads, measure


def zipup(shapes, i):
    weights = []
    for j in shapes:
        length = prod(j)
        layer, i = i[:length], i[length:]
        weights.append(layer.reshape(j))
    return weights


# def login():
#     td_session = TDClient(
#         client_id='5WINVND6ZU0XRIELK5DRJLHZGK9KGYGB',
#         redirect_uri='http://localhost',
#         credentials_path='C:\\Users\\chinm\\PycharmProjects\\StockTracking\\td_cred.json'
#     )
#
#     # Login to the session
#     td_session.login()
#     return td_session
#
#
# def gather_data(td_session, tic, k=4):
#     table = pd.DataFrame()
#     # for tic in args:
#     hist_data = td_session.get_price_history(symbol=str(tic), period_type='year', period='10', frequency_type='daily',
#                                              frequency='1', extended_hours=False)
#     #     print(hist_data.keys())
#     #     print(not hist_data['empty'])
#     #
#     if hist_data is None:
#         print('no data was acquired from td ameritrade')
#         exit(0)
#     if not hist_data['empty']:
#         table = pd.DataFrame(hist_data['candles'])
#     # print(json.dumps(hist_data['candles'], indent=4))
#     table.pop('volume')
#     table.pop('datetime')
#     mat = table.to_numpy()
#
#     x = np.array([np.zeros(shape=(4, 4))])
#     y = np.array([np.zeros(shape=(4, 4))])
#
#     print(type(len(mat)))
#     print(k)
#     for i in range(len(mat) - k - 1):
#         x = np.append(x, np.array([mat[i:i + k, :]]), axis=0)
#         y = np.append(y, np.array([mat[i + 1:i + 1 + k, :]]), axis=0)
#     x = np.delete(x, 0, axis=0)
#     y = np.delete(y, 0, axis=0)
#     return x, y
#

def split_data(x, y, test_percent=.15, validation_percent=.15):
    test_len = int(len(x) * test_percent)
    val_len = int(len(x) * validation_percent)
    # shuffle = np.random.permutation(len(x))
    # x = x[shuffle]
    # y = y[shuffle]
    x_val = x[-val_len:, :, :]
    y_val = y[-val_len:, :, :]

    x_test = x[-test_len - val_len:-val_len, :, :]
    y_test = y[-test_len - val_len:-val_len, :, :]

    x_train = x[:-test_len - val_len, :, :]
    y_train = y[:-test_len - val_len, :, :]

    return x_train, y_train[:, 0, :], x_test, y_test[:, 0, :], x_val, y_val[:, 0, :]


def unzip(grad):
    arr = np.array([])
    for i in grad:
        arr = np.append(arr, i.numpy())
    return arr
