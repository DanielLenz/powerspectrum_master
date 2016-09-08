# Monte Carlo Apodised Spherical Transform EstimatoR

from __future__ import division
import numpy as np
import itertools as it
import healpy as hp
from scipy import linalg
from joblib import Memory

from .utilities import wigner_3j_sq

memory = Memory(cachedir='.joblib/', verbose=False)


class PowSpecEstimator(object):
    """docstring for PowSpec"""

    _cl_conv = None
    _cl_deconv = None
    _cl_mask = None
    _nside = None
    _ls = None

    _pixfunc = None
    _beamfunc1 = None
    _beamfunc2 = None
    _windowfunc1 = None
    _windowfunc2 = None

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
            beam1=None,
            beam2=None):

        self._map1 = map1
        self._map2 = map2
        self._mask = mask

        if lmax is None:
            self.lmax = self.nside
        else:
            self.lmax = lmax

        self.beam1 = beam1
        self.beam2 = beam2

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
    def map1(self):
        return self._map1

    @property
    def map2(self):
        if self._map2 is None:
            self._map2 = self.map1
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
                self.map1*self.mask,
                self.map2*self.mask,
                lmax=self.lmax-1)
        return self._cl_conv

    @property
    def cl_mask(self):
        if self._cl_mask is None:
            self._cl_mask = self.get_cl_conv(
                self.mask,
                self.mask,
                lmax=self.lmax-1)
        return self._cl_mask

    @property
    def pixfunc(self):
        if self._pixfunc is None:
            self._pixfunc = hp.pixwin(self.nside, pol=False)[:self.lmax]
        return self._pixfunc

    @property
    def beamfunc1(self):
        if hasattr(self.beam1, '__iter__'):
            self._beamfunc1 = self.beam1
        else:
            self._beamfunc1 = hp.gauss_beam(
                np.deg2rad(self.beam1),
                lmax=self.lmax-1,
                pol=False)
        return self._beamfunc1

    @property
    def beamfunc2(self):
        # if self.beam2 is None and self._map2 is None:
        #     self._beamfunc2 = self.beamfunc1
        #     return self._beamfunc2
        if self.beam2 is None:
            self._beamfunc2 = np.ones(self.lmax, dtype=np.float32)

        elif hasattr(self.beam2, '__iter__'):
            self._beamfunc2 = self.beam2
        else:
            self._beamfunc2 = hp.gauss_beam(
                np.deg2rad(self.beam2),
                lmax=self.lmax-1,
                pol=False)
        return self._beamfunc2

    @property
    def windowfunc1(self):
        self._windowfunc1 = self.beamfunc1 * self.pixfunc
        return self._windowfunc1

    @property
    def windowfunc2(self):
        self._windowfunc2 = self.beamfunc2 * self.pixfunc
        return self._windowfunc2


    @property
    def cl_conv_b(self):
        _cl_conv_b = self.bin_spectrum(self.cl_conv, binfactor=self.binfactor)
        return _cl_conv_b

    @property
    def cl_deconv(self):
        if self._cl_deconv is None:
            self._cl_deconv = (
                np.linalg.lstsq(self.M_l1l2, self.cl_conv)[0] /
                self.windowfunc1*self.windowfunc2)
        return self._cl_deconv

    @property
    def cl_deconv_b(self):
        _cl_deconv_b = self.bin_spectrum(
            self.cl_deconv,
            binfactor=self.binfactor)
        return _cl_deconv_b

    @property
    def M_l1l2(self):
        if self._M_l1l2 is None:
            self._M_l1l2 = determine_M_l1l2(self.lmax, self.cl_mask)
        return self._M_l1l2

    # Class functions
    ##########################################
    def get_cl_conv(self, map1, map2, lmax):
        isfinite = np.isfinite(map1) & np.isfinite(map2)
        map1 = np.where(isfinite, map1, hp.UNSEEN)
        map2 = np.where(isfinite, map2, hp.UNSEEN)
        cl_conv = hp.anafast(map1, map2, lmax=lmax, iter=3)
        return cl_conv

    def set_binfactor(self, binfactor):
        self.binfactor = binfactor
        self.bin_centres = self.bin_spectrum(
            np.arange(self.cl_conv.shape[0]),
            binfactor)

    def bin_spectrum(self, spectrum, binfactor):
        spectrum_binned = np.mean(spectrum.reshape((-1, binfactor)), axis=1)
        return spectrum_binned


# Functions
##########################################
@memory.cache
def determine_M_l1l2(lmax, cl_mask):
    M_l1l2 = np.zeros((lmax, lmax), dtype=np.float32)

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
