# Monte Carlo Apodised Spherical Transform EstimatoR

from __future__ import division
import numpy as np
from numba import njit
from sympy.physics import wigner
import itertools as it


def wignersq_exact(l1, l2, l3):
    result = wigner.wigner_3j(l1, l2, l3, 0, 0, 0)
    return result**2


def wignersq_simple(l1, l2, l3):
    L = l1 + l2 + l3

    a1 = np.long(np.math.factorial(L-2*l1))
    a2 = np.long(np.math.factorial(L-2*l2))
    a3 = np.long(np.math.factorial(L-2*l3))
    a4 = np.long(np.math.factorial(L+1))

    term1 = np.sqrt(a1 * a2 * a3 / a4)

    b1 = np.math.factorial(L/2)
    b2 = np.math.factorial(L/2-l1)
    b3 = np.math.factorial(L/2-l2)
    b4 = np.math.factorial(L/2-l3)

    term2 = (b1 / b2 / b3 / b4)

    out = (-1)**(L/2) * term1 * term2
    return out*out


@njit
def wignersq_approx(l1, l2, l3):

    a1 = 2*np.power(l1*l2, 2)
    a2 = 2*np.power(l1*l3, 2)
    a3 = 2*np.power(l2*l3, 2)
    b1 = np.power(l1, 4)
    b2 = np.power(l2, 4)
    b3 = np.power(l3, 4)

    term1 = a1 + a2 + a3 - b1 - b2 - b3
    return 2./np.pi*term1**(-0.5)


def make_M_l1l2(ls, W):
    M_l1l2 = np.zeros((ls.size, ls.size), dtype=np.float32)
    for l1, l2, l3 in it.product(ls, repeat=3):
        L = l1 + l2 + l3
        if L % 2:
            continue
        if (np.abs(l1-l2) > l3) or (l3 > (l1 + l2)):
            continue
        factor = (2. * l2 + 1.) / 4. / np.pi
        # if L < 64:
        wigner_term = (
            (2. * l3 + 1.) * W[l3-ls[0]] *
            wignersq_simple(l1, l2, l3))
        # else:
        #     wigner_term = (2 * l3 + 1) * W[l3] * wignersq_approx(l1, l2, l3)

        M_l1l2[l1-ls[0], l2-ls[0]] += factor * wigner_term
    return M_l1l2


def make_P_bl(ls, nbins):
    P_bl = np.zeros((nbins, ls.shape[0]))
    l_lows = np.linspace(ls[0], ls[-1], nbins+1, dtype=np.int)
    bin_centres = np.diff(l_lows)/2 + l_lows[:-1]

    for b, l in it.product(np.arange(nbins), ls-ls[0]):
        if (2 <= l_lows[b]) & (l_lows[b] <= l) & (l < l_lows[b+1]):
            P_bl[b, l] += 1./2./np.pi * l * (l+1.) / (l_lows[b+1] - l_lows[b])
    return P_bl, bin_centres


def make_Q_lb(ls, nbins):
    Q_lb = np.zeros((ls.shape[0], nbins))
    l_lows = np.linspace(ls[0], ls[-1], nbins+1, dtype=np.int)
    bin_centres = np.diff(l_lows)/2 + l_lows[:-1]

    for b, l in it.product(np.arange(nbins), ls-ls[0]):
        if (2 <= l_lows[b]) & (l_lows[b] <= l) & (l < l_lows[b+1]):
            Q_lb[l, b] += 2.*np.pi / l / (l+1.)

    return Q_lb, bin_centres
































