import numpy as np
import matplotlib.pyplot as plt

def L(t, data):
    rval = 1.0
    for elem in data:
        if elem == 1:
            rval *= t
        elif elem == 0:
            rval *= (1 - t)
        else:
            print("bad")
            exit()
    return rval

def Lprime(t, data):
    rval = 1.0
    for elem in data:
        rval *= (t**elem)
        rval *= ((1-t)**(1-elem))
    return rval

def main():
    data = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

    X = np.linspace(0.0, 1.0, 101)
    y = []
    
    for elem in X:
        y.append(L(elem, data))

    max = 0
    maxind = 0
    for index, elem in enumerate(y):
        if elem > max:
            maxind = index
            max = elem

    plt.plot(X, y)

    plt.axvline(X[maxind], color='r')

    plt.xlabel("theta")
    plt.ylabel("L(theta)")
    plt.show()

    data = [1]*3 + [0]*2

    X = np.linspace(0.0, 1.0, 101)
    y = []
    
    for elem in X:
        y.append(L(elem, data))

    max = 0
    maxind = 0
    for index, elem in enumerate(y):
        if elem > max:
            maxind = index
            max = elem

    plt.plot(X, y)

    plt.axvline(X[maxind], color='r')

    plt.xlabel("theta")
    plt.ylabel("L(theta)")
    plt.show()

    data = [1]*60 + [0]*40

    X = np.linspace(0.0, 1.0, 101)
    y = []
    
    for elem in X:
        y.append(L(elem, data))

    max = 0
    maxind = 0
    for index, elem in enumerate(y):
        if elem > max:
            maxind = index
            max = elem

    print max

    plt.plot(X, y)

    plt.axvline(X[maxind], color='r')

    plt.xlabel("theta")
    plt.ylabel("L(theta)")
    plt.show()

    data = [1]*5 + [0]*5

    X = np.linspace(0.0, 1.0, 101)
    y = []
    
    for elem in X:
        y.append(L(elem, data))

    max = 0
    maxind = 0
    for index, elem in enumerate(y):
        if elem > max:
            maxind = index
            max = elem

    print max

    plt.plot(X, y)

    plt.axvline(X[maxind], color='r')

    plt.xlabel("theta")
    plt.ylabel("L(theta)")
    plt.show()

if __name__ == "__main__":
    main()
