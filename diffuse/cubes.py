"""

"""
import os, sys
import shutil
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import healpy
from astropy.io import fits

from utilities import SkyDir, HealpixCube, ait_plot

from jupydoc import DocPublisher

__docs__ = ['FactorCube', 'ApplyToDiffuse']

bubble_path = '/mnt/c/Users/thbur/OneDrive/fermi/bubbles/'

class FactorCube(DocPublisher):
    """
    title: Galactic Diffuse Factor 
    author: Toby Burnett
    
    sections: load_files 
    
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_files(self):
        """

        """
        self.publishme()


class ApplyToDiffuse(DocPublisher):
    """
    title: Application of the factor cube to the diffuse model
    author: Toby Burnett
    sections: galactic_diffuse_model correction_factor_maps apply_factors

    diffuse_file: 'test_model_InnerGalaxyYB01_test512_interp_nopatch.fits'
    factor_cubic_fit_file: 
    diffuse_path: '/mnt/c/Users/thbur/Onedrive/fermi/diffuse/'
    bubble_path: '/mnt/c/Users/thbur/Onedrive/fermi/bubbles/'
    factor_file: 'bubble_cube_v4.fits'
    diffuse_model_name: 'YB01'
    """
    def galactic_diffuse_model(self):
        """The Diffuse Model

        The Galactic diffuse model used for the 4FGL catalog is described in the paper.

        This note describes the application of my "factor cube" to a galactic diffuse model {self.diffuse_model_name} without its "patch", 
        defined by the file `{self.diffuse_file}`.

        {fig1}

        """
        self.dm = dm= HealpixCube(os.path.join(self.diffuse_path, self.diffuse_file))
        bunit = dm.bunit.replace(' ', '\\ ')
        fig1 =plt.figure(figsize=(12,5), num=1)
        fig1.caption = 'Flux for Galactic diffuse model "InnerGalaxyYB01" without patch, at 1 GeV'
        ait_plot(dm, 1000., title='1 GeV', fig =fig1, log=True, cb_kw=dict(label=fr'$\mathrm{{Flux\ ({bunit})}}$'))
     
        #---------------
        self.publishme()

    def correction_factor_maps(self):
        """Correction Factor maps

        I described the procedure to apply corrections in this 
        [section of my 4FGL announcement](https://confluence.slac.stanford.edu/display/~burnett/4FGL+source+list+announcement#id-4FGLsourcelistannouncement-TheXC02_v3model). 
        This used the earlier XC02_v3 version. I'll assume that the later one used here, {self.diffuse_model_name}, would have similar corrections, but more reliable galactic features.

        My procedure generated eight maps from 100 MeV to 10 GeV. Here is the (interpolated) factor that would apply to 1 GeV:

        {fig2}

        To apply these corrections to the diffuse cube, I need to interpolate and extrapolate. I'll use the log-parabola fits that I made to the above file for this purpose. For most of the sky, the fit is quite good. (A notable exception is north of the anti-center, a subject for another day perhaps.)

        A slight caveat is that at the south bubble, such an extrapolation might be a large over prediction: Here is the fit compared with the measurments at $(l,b) = (0,-50)$:
        {fig3}
        (Is the apparent excess above the fit for the last bin real?)
        So for now, I'll cap the correction facctor at 5.0.
        """
        fig2 =plt.figure(figsize=(10,4), num=2)
        
        dm= HealpixCube(os.path.join(self.bubble_path, self.factor_file))
        ait_plot(dm, 1000., title='1 GeV', fig=fig2, cb_kw=dict(label='correction factor') )
        fig2.caption='Derived correction factor map at 1 GeV'

        pf = Polyfit(bubble_path+'/bubble_cube_v4.fits', bubble_path+'/bubble_sigmas.fits')
        fig3, ax = plt.subplots(figsize=(4,4), num=3)
        pf.plot_fit(0,-50, ax=ax);
        self.pf = pf
        #---------------
        self.publishme()

    def apply_factors(self):
        """Application of the factor cube
        """
        #---------------
        self.publishme()


class Polyfit(object):
    """ Manage a log parabola fit to every pixel"""
    def __init__(self, cubefile, sigsfile, start=0, stop=8, deg=2):
        
        m = HealpixCube(cubefile)
        msig = HealpixCube(sigsfile)
        
        meas = np.array([m[i] for i in range(8)])
        sig  = np.array([msig[i] for i in range(8)])

        self.planes = np.array(range(start,stop)) # plane numbers
        self.values = meas[start:,:]
        weights = 100./sig[start:,:] #from percent
        self.wtmean = weights.mean(axis=1)

        self.fit, self.residuals, self.rank, self.svals, self.rcond =\
            np.polyfit(self.planes,self.values, deg=deg, full=True, w=self.wtmean)
            
        labels= 'intercept slope curvature'.split()   
        
    def __getitem__(self, i):
        return self.fit[i]
    
#     def ait_plots(self):
#         self.hpfit=[healpix_map.HParray(labels[deg-i], self.fit[i,:]) for i in range(deg,-1,-1)]
#         healpix_map.multi_ait(self.hpfit, cmap=plt.get_cmap('jet'),  grid_color='grey')

    def __call__(self, x, n):
        # if not hasattr(x, '__iter__'):
        #     x = np.array([x])
        x = np.atleast_1d(x)
        fit= self.fit[:,n]; 
        t =fit.reshape(3,1)
        return ( t * np.vstack([x**2, x, np.ones(len(x))] )).sum(axis=0)
    
    def ang2pix(self, glon, glat):
        return healpy.ang2pix(64, glon, glat, lonlat=True)
        
    def get_fit(self, pix):
             
        y = self.values[:,pix]
        yerr = 1/self.wtmean
        fn = lambda xx : self(xx, pix)
        return y, yerr, fn
    
    def plot_fit(self, glon, glat, ax=None, axis_labels=True):
        pix = self.ang2pix(glon, glat)
        y, yerr, fn = self.get_fit(pix)

        fig, ax =plt.subplots() if ax is None else (ax.figure, ax)
        npl = len(self.planes)
        xx = np.linspace(self.planes[0]-0.5,self.planes[-1]+0.5,2*npl+1)

        ax.errorbar(self.planes, y, yerr=yerr, fmt='o', ms=8);
        ax.plot(xx, fn(xx), '-', lw=2);
        ax.text(0.05,0.9,'({:3.0f}, {:+2.0f})'.format(glon, glat), transform=ax.transAxes)
        if axis_labels:
            ax.set(ylabel='flux factor', xlabel='Energy (GeV)')
            ax.set(xticks=[-0.5, 3.5, 7.5], xticklabels='0.1 1.0 10'.split())
        else:
            ax.set_xticks(self.planes[::2])  
        ax.axhline(1.0, color='grey')
        ax.grid(alpha=0.3)
        
    
    def multiplot(self, glons, glats, grid_shape=(4,5), title=''):
 
        fig, axx = plt.subplots(grid_shape[0],grid_shape[1], figsize=(12,12), sharex=True, sharey=True,
                            gridspec_kw=dict(left=0.05, right = 0.95,top=0.95, wspace=0, hspace=0)  )
        for glon, glat, ax in zip(glons, glats, axx.flatten()):
            self.plot_fit( glon, glat, ax, axis_labels=False)
        fig.suptitle(title, fontsize=16); 
        fig.set_facecolor('white')
           