import numpy as np
import matplotlib.pyplot as plt
import math

# from Distances import euclidean
# from typing import Callable
# from numba import jit

# p = np.array([[0.2, 2.0],
#               [1.5, 2.8],
#               [2.3, 1.6],
#               [2.9, 1.8],
#               [4.1, 3.1],
#               [5.6, 2.9],
#               [7.2, 1.3],
#               [8.2, 1.1]])
#
# q = np.array([[0.3, 1.6],
#               [3.2, 3.0],
#               [3.8, 1.8],
#               [5.2, 3.1],
#               [6.5, 2.8],
#               [7.0, 0.8],
#               [8.9, 0.6]])

p, q = [], []


# @jit(nopython=True)
def calculate(ca: np.ndarray, i: int, j: int) -> float:
    """
    Calculates the distance between p[i] and q[i]
    :param i: Index into poly-line p
    :param j: Index into poly-line q
    :return: Distance value
    """
    if ca[i, j] > -1.0:
        # Uncomment the line below to see when the code does the dynamic programming trick: reuses already-calculated
        # values
        # print(i, j, "*")
        return ca[i, j]

    # Uncomment the line below to follow the order of recursive calls
    # print(i, j)
    # Distances.py library is not working here, manual calculation required
    # d = dist_func(p[i].tolist(), q[j].tolist())
    d = math.sqrt((p[i][0] - q[j][0])**2 + (p[i][1] - q[j][1])**2)

    if i > 0 and j > 0:
        ca[i, j] = max(min(calculate(ca, i-1, j),
                           calculate(ca, i-1, j-1),
                           calculate(ca, i, j-1)), d)
    elif i > 0 and j == 0:
        ca[i, j] = max(calculate(ca, i-1, 0), d)
    elif i == 0 and j > 0:
        ca[i, j] = max(calculate(ca, 0, j-1), d)
    else:
        ca[i, j] = d

    # Uncomment the line below to follow the return order of the calculated values.
    # print(i, j)
    # This is how the order of the returned coordinates was calculated in the Medium article.
    return ca[i, j]


# @jit(nopython=True)
def recursive_frechet_calculator(p: np.ndarray, q: np.ndarray) -> (float, np.ndarray):
    """
    Calculates the Fréchet distance between poly-lines p and q
    This function implements the algorithm described by Eiter & Mannila
    :param p: Poly-line p
    :param q: Poly-line q
    :return: Distance value
    """
    n_p = p.shape[0]
    n_q = q.shape[0]
    ca = np.zeros((n_p, n_q), dtype=np.float64)
    ca.fill(-1.0)
    return calculate(ca, n_p - 1, n_q - 1), ca


# @jit(nopython=True)
def recursive_frechet(p: np.ndarray, q: np.ndarray) -> float:
    d, ca = recursive_frechet_calculator(p, q)
    return d


# @jit(nopython=True)
def recursive_frechet_diag(p: np.ndarray, q: np.ndarray) -> float:
    d, ca = recursive_frechet_calculator(p, q)
    return ca


def linear_frechet(pi: np.ndarray, qi: np.ndarray) -> float:
    p = pi
    q = qi

    n_p = p.shape[0]
    n_q = q.shape[0]
    ca = np.zeros((n_p, n_q), dtype=np.float64)

    for i in range(n_p):
        for j in range(n_q):
            # d = dist_func(p[i], q[j])
            d = math.sqrt((p[i][0] - q[j][0]) ** 2 + (p[i][1] - q[j][1]) ** 2)

            if i > 0 and j > 0:
                ca[i, j] = max(min(ca[i - 1, j],
                                   ca[i - 1, j - 1],
                                   ca[i, j - 1]), d)
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], d)
            else:
                ca[i, j] = d
    return ca[n_p - 1, n_q - 1]


def generate_pattern(l: int, t: str, f: float, n: int) -> list:
    """
    Generates patterns according to input
    :param l: length of the output pattern
    :param t: what pattern (full sin, half positive sin, half negative sin etc.)
    :param f: factor by which pattern is modified
    :param n: number of repetitions
    :return: list of floats with given pattern
    """

    assert l > 0, "Pattern generator: length must be greater than zero"
    res = []

    if t == 'full sin':
        for i in range(0, l):
            res.append(f * (1 + math.sin(((i / l) * n) * 2 * math.pi - math.pi/2))/2)
    elif t == 'half sin':
        for i in range(0, l):
            res.append(f * abs(math.sin(((i / l) * n) * math.pi)))

    return res