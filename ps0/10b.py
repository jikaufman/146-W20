import numpy as np
import matplotlib.pyplot as plt



def main():
    mean = [1, 1]
    cov = [[1, 0], [0, 1]]

    x, y = np.random.multivariate_normal(mean, cov, 1000).T
    plt.plot(x, y, 'x')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()
