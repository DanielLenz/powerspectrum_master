import numpy as np
import numba
import itertools as it

logfactorials = np.array(
    [np.sum(np.log(np.arange(1, n+1))) for n in range(4096)])


@numba.njit()
def wigner_3j_sq(l1, l2, l3):
    L = l1+l2+l3
    term1 = (
        logfactorials[L-2*l1] + logfactorials[L-2*l2] +
        logfactorials[L-2*l3] - logfactorials[L+1])
    term2 = (
        logfactorials[L/2] - logfactorials[L/2-l1] -
        logfactorials[L/2-l2] - logfactorials[L/2-l3])

    return np.exp(term1 + 2*term2)


def precompute_wigner_3j(lmax=1024, outpath='resources/wigner_3j'):
    w_l1l2l3 = np.zeros((lmax, lmax, lmax))
    for l1, l2, l3 in it.product(np.arange(lmax), repeat=3):
        L = l1+l2+l3
        if L % 2:
            continue
        if (np.abs(l1-l2) > l3) or (l3 > (l1 + l2)):
            continue
        w_l1l2l3[l1, l2, l3] += wigner_3j_sq(l1, l2, l3)

    np.save('outpath', w_l1l2l3, allow_pickle=False)

    return 0
