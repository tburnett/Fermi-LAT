"""

"""
import os, sys
import shutil
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import healpy
from astropy.io import fits

from utilities import SkyDir, HPmap, HPcube, HPratio, ait_plot

from jupydoc import DocPublisher

__docs__ = ['FactorCube', 'DiffuseFactorCube']

bubble_path = '/home/burnett/fermi/bubbles/'

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


class DiffuseFactorCube(DocPublisher):
    """
    title: Application of the factor cube to the diffuse model
    author: Toby Burnett
    sections: galactic_diffuse_model flux_factor_cube [dm7_comparison] apply_factors

    fermi_path: '/home/burnett/fermi'
    diffuse_file: 'diffuse/test_model_InnerGalaxyYB01_test512_interp_nopatch.fits'
    factor_file: 'bubbles/bubble_cube_v4.fits'
    test_model_name: 'YB01_nopatch'
    diffuse_model_name: 'gll_iem_v07'
    diffuse_v07: 'diffuse/gll_iem_v07_hpx.fits'
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def galactic_diffuse_model(self):
        """The Diffuse Model

        The Galactic diffuse model {self.diffuse_model_name} used for the 4FGL catalog is described in the paper.

        This note describes the application of my "factor cube" to the special galactic diffuse model {self.test_model_name}, 
        defined by the file `{self.diffuse_file}`.

        {fig1}

        """
        self.dm = dm= HPcube.from_FITS(os.path.join(self.fermi_path, self.diffuse_file))
        
        fig1 =plt.figure(figsize=(12,5), num=1)
        fig1.caption = f'Flux for Galactic diffuse test model "{self.test_model_name}" without patch, at 1 GeV'
        ait_plot(dm, 1000., label='1 GeV', fig =fig1, log=True,)
     
        #---------------
        self.publishme()

    def flux_factor_cube(self):
        """The Flux Factor Cube

        
        I described the procedure to apply corrections in this 
        [section of my 4FGL announcement](https://confluence.slac.stanford.edu/display/~burnett/4FGL+source+list+announcement#id-4FGLsourcelistannouncement-TheXC02_v3model). A further of examination of the properties of quadratic fits to the eight planes is [here](https://confluence.slac.stanford.edu/display/~burnett/2018/11/30/A+more+detailed+look+at+the+XC02+flux+factor+maps).
        This used the earlier test XC02_v3 version. I'll assume that the final one, {self.diffuse_model_name}, would have similar corrections, but more reliable galactic features.

        My procedure generated eight maps for the energy bands from 100 MeV to 10 GeV. Here is the (interpolated) factor that would apply to 1 GeV:

        {fig2}

        To apply these corrections to the diffuse cube, I need to interpolate and extrapolate. I'll use the log-parabola fits that I made to the above file for this purpose. For most of the sky, the fit is quite good. (A notable exception is north of the anti-center, a subject for another day perhaps.)

        A  caveat is that at the south bubble, such an extrapolation might be a large over prediction: Here is the fit compared with the measurements at $(l,b) = (0,-50)$:
        {fig3}
        (Is the apparent excess above the fit for the last bin real?)
        So for now, I'll impose the limits {limits}. Here is the extrapolation to 1 TeV:
        {fig4}

        """
        limits = (0.5, 25)
        fig2 =plt.figure(figsize=(7,3), num=2)
        
        dm= HPcube.from_FITS(os.path.join(self.fermi_path, self.factor_file))
        ait_plot(dm, 1000., fig=fig2, label='1 GeV', cblabel= 'flux factor')
        fig2.caption='Interpolated factor factor map at 1 GeV'

        # Createe the PolyFit object
        pf = Polyfit(self.fermi_path+'/bubbles/bubble_cube_v4.fits',
                     self.fermi_path+'/bubbles/bubble_sigmas.fits', 
                     limits=limits)
        fig3, ax3 = plt.subplots(figsize=(4,3), num=3)
        pf.plot_fit(0,-50, ax=ax3)

        fig4 = plt.figure(figsize=(7,3), num=4)
        ait_plot(pf, 1e6, fig=fig4, label='1 TeV', cblabel='flux factor',  )
        fig4.caption='Extrapolated flux factor map at 1 TeV'
        self.pf = pf
        #---------------
        self.publishme()

    def dm7_comparison(self):
        """Comparison with the DM7 model

        The "factor cube" is equivalent to the "patch" component of the diffuse model, except of
        course the latter does not cover the whole sky, and cannot account for an overprediction 
        since it is constrained to represent a positive flux. After the fact, I can derive the
        current equivalent by dividing the released model DM7 by the test model {self.test_model_name}.

        Here it is at 1 GeV.
        {fig1}
        """
        E, label= 1000., '1 GeV'
        self.dm7 = HPcube.from_FITS(os.path.join(self.fermi_path, self.diffuse_v07))
  

        # dm7a = self.dm7.hpmap(E, label=f'DM7 @ {label}')
        # ratio = dm7a.map / self.dm.hpmap(E).map; 
        # hpratio = HPmap(ratio, f'ratio @ {label}')
        fig1 = plt.figure(figsize=(7,3), num=1)
        hpratio = HPratio(self.dm7, self.dm)
        hpratio.ait_plot( E, label=label, fig=fig1)
        fig1.caption=f'Ratio of diffuse model v07 to the test model {self.test_model_name}'

        #---------------
        self.publishme()

    def apply_factors(self):
        """Application of the factor cube

        The class which performs the quadratic fit to the individual planes implements what I call a "sky function". 
        That is, it defines a member function `def __call__(coord, energy)` where:
         * `coord` is an object of my class `SkyDir`. It wraps an instance of an `astropy.coords.SkyCoord`, but is more convenient. A big difference with the fermipy version is that is supports multiple coordinate values, and converstion to/from HEALPix.
         * `energy` is one or more energies in MeV. 


         Both the class that interprets a FITS HEALPix cube, and the factor cube described in the last section implement this 
         interface. All-sky plots are easily generated by passing one of these objects, and an energy, to my `ait_plot`.



        """
        #---------------
        self.publishme()



class Polyfit(object):
    """ Manage a log parabola fit to every pixel"""
    def __init__(self, cubefile, sigsfile, start=0, stop=8, deg=2,limits=(0.5,25)):
        """
        """
        
        m = HPcube.from_FITS(cubefile)
        msig = HPcube.from_FITS(sigsfile)
        self.nside= m.nside
        self.limits=limits
        
        meas = np.array([m[i].map for i in range(8)])
        sig  = np.array([msig[i].map for i in range(8)])

        self.planes = np.array(range(start,stop)) # plane numbers
        self.values = meas[start:,:]
        weights = 100./sig[start:,:] #from percent
        self.wtmean = weights.mean(axis=1)

        self.fit, self.residuals, self.rank, self.svals, self.rcond =\
            np.polyfit(self.planes,self.values, deg=deg, full=True, w=self.wtmean)
            
        labels= 'intercept slope curvature'.split()   
        
    def __getitem__(self, i):
        return self.fit[i]

    def energy_index(self, energy):
        """convert energy in MeV to correspond to np.logspace(2.125, 3.875, 8)
        """
        energy = np.atleast_1d(energy)
        return 4*np.log10(energy/100)-0.5
    
#     def ait_plots(self):
#         self.hpfit=[healpix_map.HParray(labels[deg-i], self.fit[i,:]) for i in range(deg,-1,-1)]
#         healpix_map.multi_ait(self.hpfit, cmap=plt.get_cmap('jet'),  grid_color='grey')

    def __call__(self, 
                coord: 'SkyDir', 
                energy:'energies in MeV',
                factor_cap=25,
                
                )->'interpolated list of factors':

        """
        Implements the sky function using the logparabola fit
        Note that either arg can be multivalued, but not both.
        """
        energy = np.atleast_1d(energy)
        pix = np.atleast_1d(coord.to_healpix(self.nside) )        
        x = self.energy_index(energy)   
        fit= self.fit[:,pix]; 
        xx= np.vstack([x**2, x, np.ones(len(x))] )
        ret = np.matmul(fit.T, xx).T
        ret = np.clip(ret, *self.limits)
        return ret[0] if ret.shape[0]==1 else ret

    def parabolic_fit(self, 
            x:'energy index', 
            pix:'pixel index'):
        """Evaluate the parabolic fit in energy index
        """
        # if not hasattr(x, '__iter__'):
        #     x = np.array([x])
        x = np.atleast_1d(x)
        fit= self.fit[:,pix]; 
        t =fit.reshape(3,1)
        return ( t * np.vstack([x**2, x, np.ones(len(x))] )).sum(axis=0)
    
    def ang2pix(self, glon, glat):
        return healpy.ang2pix(64, glon, glat, lonlat=True)
        
    def get_fit(self, pix):
             
        y = self.values[:,pix]
        yerr = 1/self.wtmean
        fn = lambda xx : self.parabolic_fit(xx, pix)
        return y, yerr, fn
    
    def plot_fit(self, glon, glat, ax=None, axis_labels=True):

        pix = self.ang2pix(glon, glat)
        y, yerr, fn = self.get_fit(pix)

        fig, ax =plt.subplots() if ax is None else (ax.figure, ax)
        npl = len(self.planes)
        xx = np.linspace(self.planes[0]-0.5,self.planes[-1]+0.5,2*npl+1)

        ax.errorbar(self.planes, y, yerr=yerr, fmt='o', ms=8, label='measured values' if axis_labels else '');
        ax.plot(xx, fn(xx), '-', lw=2, label='parabolic fit' if axis_labels else '');
        ax.text(0.05,0.9,'({:3.0f}, {:+2.0f})'.format(glon, glat), transform=ax.transAxes)
        if axis_labels:
            ax.set(ylabel='flux factor', xlabel='Energy (GeV)')
            ax.set(xticks=[-0.5, 3.5, 7.5], xticklabels='0.1 1.0 10'.split())
            ax.legend()
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
           