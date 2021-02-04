import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def main():
    a = np.array([[1, 0], [1, 3]])
    w, v = LA.eig(a)
    print w
    print v


if __name__ == "__main__":
    main()
