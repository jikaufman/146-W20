import numpy as np
import matplotlib.pyplot as plt



def main():
    mean = [0, 0]
    cov = [[2, 0], [0, 2]]

    x, y = np.random.multivariate_normal(mean, cov, 1000).T
    plt.plot(x, y, 'x')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()
