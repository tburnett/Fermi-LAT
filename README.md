# Fermi-LAT
The LAT, for [Large Area Telescope](https://www-glast.stanford.edu/), is the principal instrument on the NASA [Fermi gamma-ray observatory](http://fermi.gsfc.nasa.gov/)

These Jupyter notebooks are associated with my _pointlike_ all-sky analysis, used to detect, make spectral measurements, and localize point sources in the gamma-ray sky.
They are maintained at the [SLAC](https://www6.slac.stanford.edu/) computer system, the home of the Fermi-LAT analysis. That is where the python analysis code also resides. It is not (yet) at github, but code can be found in the [cvsweb](http://glast.stanford.edu/cgi-bin/cvsweb-SLAC/pointlike/python/uw/like2/) view of the repository.

The organization of the folders:

* [dev](https://github.com/tburnett/Fermi-LAT/tree/master/dev) Where I'm putting development notebooks
* [pass8](https://github.com/tburnett/Fermi-LAT/tree/master/pass8) The 6-year skymodel, used as a basis for the monthly
* [pipeline](https://github.com/tburnett/Fermi-LAT/tree/master/pipeline) Development and testing for pipeline-related issues
* [pointlike\_document](https://github.com/tburnett/Fermi-LAT/tree/master/pointlike_document) Notebooks documenting the basic design of _like2_, the new version of pointlike
* [transients](https://github.com/tburnett/Fermi-LAT/tree/master/transients) The monthly analysis: details of the 72 models of the sky constructed from the 6-year model, plus individual source finding

Note than many notebooks are works-in-progress, some forgotten. 
