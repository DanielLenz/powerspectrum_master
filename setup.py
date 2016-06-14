from setuptools import setup

setup(
    name='powspecpy',
    version='0.2',
    description='Deconvolving and binning of angular power spectra',
    author='Daniel Lenz',
    url='https://www.github.com/DanielLenz/powspecpy',
    author_email='mail@daniellenz.org',
    packages=['powspecpy', ],
    install_requires=[
        'numpy>=1.10',
        'healpy>=1.9',
        'astropy>=1.0']
)
