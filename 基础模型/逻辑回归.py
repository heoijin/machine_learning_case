import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    x = np.arange(-10, 10, 0.2)  # -10-10的数组，间隔为0.2数值
    # 非调库实现方法
    y = [sigmoid(i) for i in x]
    plt.grid(True)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()
