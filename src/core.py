# Monte Carlo Apodised Spherical Transform EstimatoR

from __future__ import division
import numpy as np
import itertools as it
from scipy import linalg


def make_M_l1l2(ls, W):
    M_l1l2 = np.zeros((ls.size, ls.size), dtype=np.float32)

    wigner_3j = np.load('resources/wigner_3j.npy', mmap_mode='r')
    for l1, l2, l3 in it.product(ls, repeat=3):
        factor = (2. * l2 + 1.) / 4. / np.pi
        wigner_term = (
            (2. * l3 + 1.) * W[l3-ls[0]] *
            wigner_3j[l1, l2, l3]
            )

        M_l1l2[l1-ls[0], l2-ls[0]] += factor * wigner_term
    return M_l1l2


def make_P_bl(ls, nbins):
    P_bl = np.zeros((nbins, ls.shape[0]))
    l_lows = np.linspace(ls[0], ls[-1], nbins+1, dtype=np.int)
    bin_centres = np.diff(l_lows)/2 + l_lows[:-1]

    for b, l in it.product(np.arange(nbins), ls):
        if (2 <= l_lows[b]) & (l_lows[b] <= l) & (l < l_lows[b+1]):
            P_bl[b, l-ls[0]] += 1./2./np.pi * l * (l+1.) / (
                l_lows[b+1] - l_lows[b])
    return P_bl, bin_centres


def make_Q_lb(ls, nbins):
    Q_lb = np.zeros((ls.shape[0], nbins))
    l_lows = np.linspace(ls[0], ls[-1], nbins+1, dtype=np.int)
    bin_centres = np.diff(l_lows)/2 + l_lows[:-1]

    for b, l in it.product(np.arange(nbins), ls):
        if (2 <= l_lows[b]) & (l_lows[b] <= l) & (l < l_lows[b+1]):
            Q_lb[l-ls[0], b] += 2.*np.pi / l / (l+1.)

    return Q_lb, bin_centres


def make_K_b1b2(ls, nbins, W, B):
    M_l1l2 = make_M_l1l2(ls, W=W)
    P_bl, bin_centres = make_P_bl(ls, nbins)
    Q_lb, _ = make_Q_lb(ls, nbins)
    K_b1b2 = np.dot(P_bl, np.dot(M_l1l2, ((B**2)[:, None] * Q_lb)))

    return K_b1b2, bin_centres


def get_C_b(ls, cl_conv, nbins, W, B):
    K_b1b2, bincentres = make_K_b1b2(ls, nbins, W, B)
    K_inv = linalg.inv(K_b1b2[1:, 1:])
    P_bl, _ = make_P_bl(ls, nbins)
    C_b = np.dot(np.dot(K_inv, P_bl[1:, 1:]), cl_conv[1:])

    return C_b, bincentres


























