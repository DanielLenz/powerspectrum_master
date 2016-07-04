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
    _cl_noise = None
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
            noise=None,
            lmax=None,
            beam=None):

        self._map1 = map1
        self._map2 = map2
        self._mask = mask
        self._noise = noise

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
    def map1(self):
        return self._map1

    @property
    def map2(self):
        return self._map2

    @property
    def noise(self):
        return self._noise

    @property
    def mask(self):
        if self._mask is None:
            self._mask = np.ones_like(self.map1, dtype=np.float32)
        return self._mask

    @property
    def cl_noise(self):
        if self.noise is None:
            raise AttributeError('No noise map was provided')
        if self._cl_noise is None:
            self._cl_noise = self.get_cl_conv(
                self.noise*self.mask, lmax=self.lmax-1)
        return self._cl_noise

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
        self._pixfunc = hp.pixwin(self.nside, pol=False)[:self.lmax]
        return self._pixfunc

    @property
    def beamfunc(self):
        if hasattr(self.beam, '__iter__'):
            self._beamfunc = self.beam
        else:
            self._beamfunc = hp.gauss_beam(
                np.deg2rad(self.beam),
                lmax=self.lmax-1,
                pol=False)
        return self._beamfunc

    @property
    def windowfunc(self):
        self._windowfunc = self.beamfunc * self.pixfunc
        return self._windowfunc

    @property
    def cl_conv_b(self):
        _cl_conv_b = self.bin_spectrum(self.cl_conv, binfactor=self.binfactor)
        return _cl_conv_b

    @property
    def cl_deconv(self):
        if self._cl_deconv is None:
            if self.noise is None:
                self._cl_deconv = (
                    np.linalg.lstsq(self.M_l1l2, self.cl_conv)[0] /
                    self.beamfunc**2)
            else:
                self._cl_deconv = (
                    np.linalg.lstsq(
                        self.M_l1l2, self.cl_conv-self.cl_noise)[0] /
                    self.beamfunc**2)
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
    def get_cl_conv(self, map1, lmax):
        map1 = np.where(np.isfinite(map1), map1, hp.UNSEEN)
        cl_conv = hp.anafast(map1, lmax=lmax, iter=3)
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
