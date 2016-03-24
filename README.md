# Powerspectrum Master
Deconvolution, binning, and estimation of uncertainties for angular power spectra

## Introduction

- *Version*: 0.1
- *Authors*: Daniel Lenz

This package allows to extract angular power spectra for cosmological background
image. It makes use of the MASTER algorithm
([Hivon et al. 2002](http://adsabs.harvard.edu/abs/2002ApJ...567....2H)).

## Installation ##

Currently, `powerspectrum_master` has no `setup` feature, so you need to clone
the repository and add it to your `PYTHONPATH`. We will change this shortly.

## Features ##

- Binning
- Deconvolving
<!-- - Caching of $M_{l_1l_2}$ matrices -->

## Dependencies ##

We kept the dependencies as minimal as possible. The following packages are
required:
* `numpy 1.10` or later
* `astropy 1.0` or later
* `healpy 1.9` or later
(Older versions of these libraries may work, but we didn't test this!)

If you want to run the notebooks yourself, you will also need the Jupyter server.

### Examples ###
Check out the [`ipython notebooks`](http://nbviewer.jupyter.org/github/DanielLenz/powerspectrum_master/blob/master/index.ipynb) in the repository for some examples of how to work with `powerspectrum_master`. Note that you only view them on the nbviewer service, and will have to clone the repository to run them on your machine.

### Who do I talk to? ###

If you encounter any problems or have questions, do not hesitate to raise an
issue or make a pull request. Moreover, you can contact the devs directly:

* <dlenz.bonn@gmail.com>
