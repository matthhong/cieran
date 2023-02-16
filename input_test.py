
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    # Show random data
    x = np.linspace(0, 10, 100)
    y = np.random.rand(100)
    plt.plot(x, y)
    plt.show()

    a = input()
    print(a)