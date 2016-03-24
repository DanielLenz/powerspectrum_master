# Monte Carlo Apodised Spherical Transform EstimatoR

from __future__ import division
import numpy as np
import itertools as it
import healpy as hp
from scipy import linalg
from joblib import Memory

from .utilities import wigner_3j_sq

memory = Memory(cachedir='.tmp/', verbose=False)


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


class PowSpecEstimator(object):
    """docstring for PowSpec"""

    _cl_conv = None
    _cl_mask = None
    _nside = None
    _ls = None

    _pixfunc = None
    _beamfunc = None
    _windowfunc = None

    _l_lows = None
    _bincentres = None

    _P_bl = None
    _Q_lb = None
    _M_l1l2 = None
    _K_b1b2 = None

    def __init__(
            self,
            map1,
            map2=None,
            mask=None,
            lmax=None,
            beam=None):

        self._map1 = map1
        self._map2 = map2
        self._mask = mask

        if lmax is None:
            self.lmax = self.nside
        else:
            self.lmax = lmax

        self.beam = beam

    # Class properties
    ##########################################
    @property
    def nside(self):
        self._nside = hp.get_nside(self.map1)
        return self._nside

    @property
    def ls(self):
        self._ls = np.arange(self.lmax)
        return self._ls

    @property
    def norm(self):
        self._norm = self.ls * (self.ls + 1.) / 2. / np.pi
        return self._norm

    @property
    def map1(self):
        return self._map1

    @property
    def map2(self):
        return self._map2

    @property
    def mask(self):
        if self._mask is None:
            self._mask = np.ones_like(self.map1, dtype=np.float32)
        return self._mask

    @property
    def cl_conv(self):
        if self._cl_conv is None:
            self._cl_conv = self.get_cl_conv(
                self.map1*self.mask, lmax=self.lmax-1)
        return self._cl_conv

    @property
    def cl_mask(self):
        if self._cl_mask is None:
            self._cl_mask = self.get_cl_conv(self.mask, lmax=self.lmax-1)
        return self._cl_mask

    @property
    def pixfunc(self):
        self._pixfunc = hp.pixwin(self.nside)[:self.lmax]
        return self._pixfunc

    @property
    def beamfunc(self):
        if hasattr(self.beam, '__iter__'):
            self._beamfunc = self.beam
        else:
            self._beamfunc = hp.gauss_beam(np.deg2rad(self.beam), self.lmax-1)
        return self._beamfunc

    @property
    def windowfunc(self):
        self._windowfunc = self.beamfunc * self.pixfunc
        return self._windowfunc

    @property
    def cl_binned(self):
        self._cl_binned = np.dot(self.P_bl, self.cl_conv)
        return self._cl_binned

    @property
    def cl_deconv(self):
        K_inv = linalg.inv(self.K_b1b2[1:, 1:])
        self._cl_deconv = np.dot(
            np.dot(K_inv, self.P_bl[1:, 1:]), self.cl_conv[1:])

        return self._cl_deconv

    @property
    def P_bl(self):
        if self._P_bl is None:
            self._P_bl = np.zeros((self.nbins, self.lmax))

            for b, l in it.product(np.arange(self.nbins), self.ls):
                if ((2 <= self.l_lows[b]) & (self.l_lows[b] <= l) &
                        (l < self.l_lows[b+1])):
                    self._P_bl[b, l] += 1./2./np.pi * l * (l+1.) / (
                        self.l_lows[b+1] - self.l_lows[b])

        return self._P_bl

    @property
    def Q_lb(self):
        if self._Q_lb is None:
            self._Q_lb = np.zeros((self.lmax, self.nbins))

            for b, l in it.product(np.arange(self.nbins), self.ls):
                if ((2 <= self.l_lows[b]) & (self.l_lows[b] <= l) &
                        (l < self.l_lows[b+1])):
                    self._Q_lb[l, b] += 2.*np.pi / l / (l+1.)

        return self._Q_lb

    @property
    def M_l1l2(self):
        if self._M_l1l2 is None:
            self._M_l1l2 = self._determine_M_l1l2(self.lmax, self.cl_mask)
        return self._M_l1l2

    @property
    def K_b1b2(self):
        if self._K_b1b2 is None:
            self._K_b1b2 = np.dot(
                self.P_bl, np.dot(
                    self.M_l1l2, ((self.windowfunc**2)[:, None] * self.Q_lb)))
        return self._K_b1b2

    # Class functions
    ##########################################
    def _determine_M_l1l2(self, lmax, cl_mask):
        M_l1l2 = np.zeros((lmax, lmax), dtype=np.float32)
        # wigner_3j = np.load('resources/wigner_3j.npy', mmap_mode='r')
        for l1, l2, l3 in it.product(np.arange(lmax), repeat=3):
            L = l1+l2+l3
            if L % 2:
                continue
            if (np.abs(l1-l2) > l3) or (l3 > (l1 + l2)):
                continue

            factor = (2. * l2 + 1.) / 4. / np.pi
            wigner_term = (
                (2. * l3 + 1.) * cl_mask[l3] * wigner_3j_sq(l1, l2, l3))

            M_l1l2[l1, l2] += factor * wigner_term

        return M_l1l2

    def get_cl_conv(self, map1, lmax):
        cl_conv = hp.anafast(map1, lmax=lmax)
        return cl_conv

    def set_bins(self, nbins):
        self.nbins = nbins
        self.l_lows = np.linspace(0, self.lmax-1, nbins+1, dtype=np.int)
        self.bin_centres = np.diff(self.l_lows)/2 + self.l_lows[:-1]















