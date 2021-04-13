from thespian.actors import *
import websocket
import numpy as np
from numpy import inf, round
import pandas as pd
from time import sleep
import ukf
import json
import requests

token = 'br78u2frh5r9l4n3ntgg'
socket = 'wss://ws.finnhub.io'


def gather_data(ticker='AAPL', back_tracking_count=4):
    table = json.dumps((requests.get(
        'https://finnhub.io/api/v1/stock/candle?symbol={}&resolution=1&from=1594639800&to=1610571600&token='
        'br78u2frh5r9l4n3ntgg'.format(
            ticker))).json())
    table = pd.read_json(table)
    table.pop('s')
    table.pop('t')
    table.pop('v')
    print(table)
    mat = table.to_numpy()

    x = np.array([np.zeros(shape=(4, 4))])
    y = np.array([np.zeros(shape=(4, 4))])

    k = back_tracking_count
    for i in range(len(mat) - k - 1):
        x = np.append(x, np.array([mat[i:i + k, :]]), axis=0)
        y = np.append(y, np.array([mat[i + 1:i + 1 + k, :]]), axis=0)
    x = np.delete(x, 0, axis=0)
    y = np.delete(y, 0, axis=0)
    return x, y


def split_data(x, y, test_percent=.25):
    test_len = int(len(x) * test_percent)
    shuffle = np.random.permutation(len(x))
    x = x[shuffle]
    y = y[shuffle]
    x_test = x[-test_len:, :, :]
    y_test = y[-test_len:, :, :]

    x_train = x[:-test_len, :, :]
    y_train = y[:-test_len, :, :]

    return x_train, y_train, x_test, y_test


class DataTable(Actor):
    def __init__(self):
        super().__init__()
        self.table = pd.DataFrame(columns=['t', 'o', 'h', 'l', 'c'])
        self.data_tup = ()

    def receiveMessage(self, message, sender):
        if message == 'close':
            print(self.table)
        else:
            self.data_tup = self.data_tup + (message)
            print(self.data_tup)
            print(len(self.data_tup))
            if len(self.data_tup) >= 5:
                print(self.data_tup)
                self.table = self.table.append(dict(map(list, self.data_tup)), ignore_index=True)
                print('appended')
                ActorSystem().tell(stop, len(self.table))
                self.data_tup = ()


class OpenClose(Actor):
    def __init__(self):
        super().__init__()
        self.time = 0
        self.open_price = 0.
        self.last_price = 0.

    def receiveMessage(self, message, sender):
        print('open_close')
        t, price = message
        t = round(t / 1000)
        if t < self.time + 60:
            self.last_price = price
        else:
            ActorSystem().tell(low, (inf))
            ActorSystem().tell(high, (-inf))
            ActorSystem().tell(data_table, (('t', self.time), ('o', self.open_price),
                                            ('c', self.last_price)))
            print('open_close data to table')
            # high.send_high()
            # low.send_low()
            self.time = t
            self.open_price = price
            # volume.vol = 0


class High(Actor):
    def __init__(self):
        super().__init__()
        self.high = -inf

    def receiveMessage(self, message, sender):
        print('high')
        price = message
        if price > self.high:
            self.high = price
        elif price == -inf:
            tup = ('h', self.high)
            ActorSystem().tell(data_table, (tup,))
            print('high data to table')
            self.high = -inf

    def send_high(self):
        ActorSystem().tell(data_table, ('h', self.high))


class Low(Actor):
    def __init__(self):
        super().__init__()
        self.low = inf

    def receiveMessage(self, message, sender):
        print('low')
        price = message
        if price < self.low:
            self.low = price
        elif price == inf:
            tup = ('l', self.low)
            ActorSystem().tell(data_table, (tup,))
            print('low data to table')
            self.low = inf


class Stream(Actor):
    def receiveMessage(self, message, sender):
        print('stream')
        data = (message['data'])[0]
        t = data['t']
        p = data['p']
        # v = data['v']
        ActorSystem().tell(open_close, (t, p))
        ActorSystem().tell(high, (p))
        ActorSystem().tell(low, (p))
        # ActorSystem().tell(volume, v)


class Stop(Actor):
    def __init__(self):
        super().__init__()

    def receiveMessage(self, msg, sender):
        if msg >= 3:
            ActorSystem().tell(web_socket, 'close')
            print('stopping')


class Socket(Actor):
    def __init__(self):
        super().__init__()
        self.ws = None

    def receiveMessage(self, msg, sender):
        if msg == 'open':
            def on_open(ws):
                ws.send('{"type":"subscribe","symbol":"TSLA"}')
                print('### OPEN ###')

            def on_message(_, message):
                message = eval(message)
                if len(message) > 1:
                    print('message')
                    print(message)
                    ActorSystem().tell(stream, message)
                    print('message sent')
                    sleep(1)

            def on_close(_):
                print('### CLOSED ###')
                print()
                ActorSystem().tell(data_table, ('close'))

            self.ws = websocket.WebSocketApp('{}?token={}'.format(socket, token), on_open=on_open,
                                             on_message=on_message,
                                             on_close=on_close)
            self.ws.run_forever()
        elif msg == 'close':
            print('### CLOSED ###')
            print()
            ActorSystem().tell(data_table, ('close'))
            self.ws.close()


def main():
    print('Creating Actors')
    stream = ActorSystem().createActor(Stream)
    data_table = ActorSystem().createActor(DataTable)
    # volume = ActorSystem().createActor(Volume)
    high = ActorSystem().createActor(High)
    low = ActorSystem().createActor(Low)
    open_close = ActorSystem().createActor(OpenClose)
    web_socket = ActorSystem().createActor(Socket)
    stop = ActorSystem().createActor(Stop)
    print('Done Creating Actors')
    ActorSystem().tell(web_socket, 'open')
    (x, y) = gather_data()
    (x_train, y_train, x_test, y_test) = split_data(x, y)


if __name__ == '__main__':
    main()
