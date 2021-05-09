import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

xdat = np.array([])
ydat = np.array([])


def animate(i):
    nonlocal xdat, ydat
    xdat = np.append(xdat, next(count()))
    ydat = np.append(ydat, np.random.randint(low=-5, high=5))

    plt.cla()
    plt.plot(xdat, ydat)

ani = FuncAnimation()
