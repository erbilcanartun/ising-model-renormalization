import numpy as np

def partial_derivative(function, variable=0, point=[]):
    arguments = point[:]
    def wraps(x):
        arguments[variable] = x
        return function(*arguments)

def lnR1(j, h):

    e1 = 2*j + 4*h
    e2 = -2*j
    emax = np.amax([e1, e2])

    return emax + np.log(np.exp(e1 - emax) + np.exp(e2 - emax))

def lnR2(j, h):

    e1 = -2*j
    e2 = 2*j - 4*h
    emax = np.amax([e1, e2])

    return emax + np.log(np.exp(e1 - emax) + np.exp(e2 - emax))

def lnR3(j, h):

    e1 = 2*h
    e2 = -2*h
    emax = np.amax([e1, e2])

    return emax + np.log(np.exp(e1 - emax) + np.exp(e2 - emax))

def J(j, h):
    return (1/4) * (lnR1(j, h) + lnR2(j, h) - 2*lnR3(j, h))

def H(j, h):
    return (1/4) * (lnR1(j, h) - lnR2(j, h))

def G(j, h):
    return (1/4) * (lnR1(j, h) + lnR2(j, h) + 2*lnR3(j, h))

def recursion_matrix(j, h):

    decimation, dimension = 2, 1
    m = decimation**dimension

    X = partial_derivative(G, 0, [j,h])
    Y = partial_derivative(J, 0, [j,h])
    Z = partial_derivative(H, 0, [j,h])

    K = partial_derivative(G, 1, [j,h])
    L = partial_derivative(J, 1, [j,h])
    M = partial_derivative(H, 1, [j,h])

    return [[m, X,   K],
            [0,     Y,   L],
            [0,     Z,   M]]

