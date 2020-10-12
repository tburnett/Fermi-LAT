"""

"""
import os, sys
import shutil, pickle
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from utilities import SkyDir, ftp

from jupydoc import DocPublisher

__docs__= ['Report']

class Report(DocPublisher):
    """
    title: UW model {skymodel} Report

    author: Toby Burnett

    sections : introduction all_sky sources

    slac_path: '/nfs/farm/g/glast/g/catalog/pointlike/skymodels/'
    local_path: '/tmp/skymodels/'

    decorator_path: https://glast-ground.slac.stanford.edu/Decorator/exp/Fermi/Decorate/groups/catalog/pointlike/skymodels/{}/plots/index.html?skipDecoration
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup()
        #self.check_slac_files()
    
    def setup(self):
        if not self.version or self.version.split('.')[-1]=='Report':
            self.skymodel = 'uw1210'
        else: self.skymodel = self.version
        mkey = self.skymodel[:4]

        year = dict(uw12='P8_12years', uw86='P305_8years', uw90='P8_10years' )[mkey]

        self.slac_path = os.path.join(self.slac_path, year,self.skymodel)
        self.local_path = os.path.join(self.local_path, self.skymodel)

        self.decorator = self.decorator_path.format(year+'/'+self.skymodel)

    def check_slac_files(self):
        with ftp.SLAC(self.slac_path, self.local_path) as slac:
            self.plot_folders = folders = filter(lambda f:f.find('.')<0, slac.listdir('plots') )
            print(f'loading/checking folders to copy from SLAC:  ', end='')
            for folder in folders:
                print(f', {folder} ', end='')
                if not os.path.isdir(os.path.join(self.local_path, folder)):
                    slac.get(f'plots/{folder}/*')
            slac.get('config.yaml'); print(', config.yaml', end='')
            slac.get('../config.yaml')
            lp = self.local_path
            shutil.move(os.path.join(lp, '../config.yaml'), 
                    os.path.join(lp, 'super_config.yaml') )
            print(', ../config.yaml -> super_config.yaml')
                

    def jpg_name(self, plot_name:'name includes path under "plots"'):
        return os.path.join(self.local_path, 'plots',  f'{plot_name}_{self.skymodel}.jpg')

    def introduction(self):
        """
        Introduction

        This a a report on the UW all-sky model {self.skymodel}. 
        I compile plots from the SLAC folder containing all its files:
        
         `{self.slac_path}`
        
        The plots are directly accessible from the  [decorator site]({self.decorator}).

        ### Configuration files: local, then superfolder

        {config}
        {super_config}
        
        {dir_info}

        """
               
        config = self.shell(f'cat {self.local_path}/config.yaml',
            summary='config.yaml' )
        
        super_config = self.shell(f'cat {self.local_path}/super_config.yaml',
            summary='../config.yaml' )
        
            
        dir_info = self.shell(f'ls -l {self.local_path}/plots', 
                summary=f' Local path ({self.local_path}/plots) contents:')

        #---------
        self.publishme()

    def all_sky(self):
        """Global, or all-sky plots

        The quality of a model can be judged by the residuals over the whole sky.
        WIth overlapping ROI's, just the ROI reeiduals are problematical, if there are 
        adjustments to the supposedly global galactic and isotropic spectra.

        Pointlike, unlike all gtlike models, allows for energy band adjustments, eight parameters rather
        than the usual three. (Isotropic is not varied.)

        {galactic_correction_maps}
        If the galactic model is reasonable, these should all be close to one, and not show features
        that should have been accounted for by the model.

        Then we look at the normalized residuals, for each ROI and energy band:

        {residual_maps}

        This shows the residuals for the four bands above 10 GeV which have no corrections. 

        The ROI's are overlapping 5-degree cones. To check the fit quality on a smaler scale, we
        look at the 1-degree scale, with nside=64 pixels. These are unique, determined by the closest
        ROI.  We show only the lowest energy band, since the statistics for the counts per pixel are
        large enough to assume the Gauusian limit for the POisson statistics, and because it is the
        most critical for source confusion.

        {residual_maps_ait}

        The distributions of the normalized residuals for high and low latitudes:
        {residual_hist}


        """
        galactic_correction_maps = self.image(self.jpg_name('environment/galactic_correction_maps'),
            width=800, caption='Galactic corrections for each ROI and the eight energy bands.')
        residual_maps = self.image(self.jpg_name('counts/residual_maps'),width=800,
            caption='ROI residual maps')
        
        residual_maps_ait = self.image(self.jpg_name('residual_maps/residual_maps_ait'),
            width=600, caption='')
        residual_hist = self.image(self.jpg_name('residual_maps/residual_hist'),width=400)

        #---------
        self.publishme()
    
    def sources(self):
        """Sources

        Each source is carefully fit to either a log-parabola or power-law with cutoff. 

        #### Cumulative TS
        {cumulative_ts}

        ### Fit quality
        {fit_quality}

        #### Low-energy fit consistency
        {spectral_fit_consistency_plots}

        #### Localization convergence
      
        The 'finish' stage of creating a model runs the localization code to check that the current position is still appropriate.
         This is measured by the change in the value of the TS at the best fit position. The position is only updated based on this
         information at the start of a new series of interations.
        {localization}

        #### Localization precision
        The association procedure records the likelihood ratio for consistency of the associated location with the fit location, expressed as a TS, or the difference in the TS for source at the maximum, and at the associated source. The distribution in this quantity should be an exponential, $\exp{{-\\Delta TS/2/f^2}}$, where $f$ is a scale factor to be measured from the distribution. If the PSF is a faithful representation of the distribution of photons from a point source, $f=1$. For 1FGL and 2FGL we assumed 1.1. The plots show the results for AGN, LAT pulsars, and all other associations. They are cut off at 9, corresponding to 95 percent containment.

        Cuts: TS>100, Delta TS<9, localization quality <5

        Applied systematic factor: 1.03 (1.37 if $|b|<5)$ and 0.45 arcmin added in quadrature with r95.

        {localization_check}



        """
        width=300
        nc_sources  = (
            ('cumulative_ts', 'Cumulative TS', 400),
            ('fit_quality', 'Fit quality. Left: Power-law fits. Tails in this distribution perhaps could be improved by changing the curvature parameter.Center: Log parabola fits.Right: Fits for the pulsars, showing high latitude subset. ', 600),
            ('spectral_fit_consistency_plots', 'Low energy spectral consistency.', 600), 
            ('pivot_vs_e0', 'Measured pivot vs. current E0.', 600),  
        )
        nc_localization = (
            ('localization', 'Left: histogram of the square root of the TS difference from current position to the fit;'\
                    ' corresponds the number of sigmas. Right: scatter plot of this vs. TS', 600),

        )
        nc_associations = (
            ('localization_check', '', 800),

        )
        locs = locals()
        def add_images(nc, analysis='sources'):
            for name, caption, width in nc:
                file = self.jpg_name(f'{analysis}/{name}')
                assert os.path.exists(file), f'{file} ?'
                image = self.image(file, width=width, caption=caption)
                locs[name] = image
        add_images(nc_sources)
        add_images(nc_localization, 'localization')
        add_images(nc_associations, 'associations')


        #---------
        self.publishme()

