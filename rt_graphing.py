import numpy as np
import matplotlib
# from itertools import count
# from matplotlib.animation import FuncAnimation
#
# plt.style.use('fivethirtyeight')
#
# xdat = np.array([])
# ydat = np.array([])
#
#
# def animate(i):
#     nonlocal xdat, ydat
#     xdat = np.append(xdat, next(count()))
#     ydat = np.append(ydat, np.random.randint(low=-5, high=5))
#
#     plt.cla()
#     plt.plot(xdat, ydat)
#
# ani = FuncAnimation()
def main():
    matplotlib.use('TKAgg')
    from matplotlib import pyplot as plt
    b = np.load('y_test.npy')[:, 0]
    vals = np.zeros_like(b)
    vals[0] = b[0]
    smoothing = 2
    for i in range(1, len(vals)):
        vals[i] = b[i]*(smoothing/11) + vals[i-1]*(1-(smoothing/11))
    plt.plot(b, label='Truth')
    plt.plot(vals, label='EMA, smooth=2')
    plt.show()

if __name__ == '__main__':
    main()
