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

    sections : introduction all_sky # test

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
            self.skymodel = 'uw1208-v3'
        else: self.skymodel = self.version
        mkey = self.skymodel[:4]

        year = dict(uw12='P8_12years', uw89='P305_8years', uw90='P8-10years' )[mkey]

        self.slac_path = os.path.join(self.slac_path, year,self.skymodel)
        self.local_path = os.path.join(self.local_path, self.skymodel)

        self.decorator = self.decorator_path.format(year+'/'+self.skymodel)

    def check_slac_files(self):
        with ftp.SLAC(self.slac_path, self.local_path) as slac:
            self.plot_folders = folders = filter(lambda f:f.find('.')<0, slac.listdir('plots') )
            print(f'loading/checking folders to copy from SLAC:  ', end='')
            for folder in folders:
                print(f'.. {folder} ', end='')
                if not os.path.isdir(os.path.join(self.local_path, folder)):
                    slac.get(f'plots/{folder}/*')
            print()

    def jpg_name(self, plot_name:'name includes path under "plots"'):
        return os.path.join(self.local_path, 'plots',  f'{plot_name}_{self.skymodel}.jpg')

    def introduction(self):
        """
        Introduction

        This a a report on the UW all-sky model {self.skymodel}. 
        I compile plots from the SLAC folder contain all its files:
        
         `{self.slac_path}`
        
        The plots are directly accessible from the 
        [decorator site](self.decorator).

        <br>Local path ({self.local_path}) contents:
        {dir_info}

        """


        dir_info = self.shell(f'ls -l {self.local_path}/plots')

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
    
    def test(self):
        """Test plots
        
        {chisq}

  
        {residual_maps_ait}
        {residual_hist}

        {}
        """
        width=600
        chisq = self.image( self.jpg_name('counts/chisq_plots'), width=400, caption='Chi squared')

        residual_maps_ait = self.image(self.jpg_name('residual_maps/residual_maps_ait'),
            width=400)
        residual_hist = self.image(self.jpg_name('residual_maps/residual_hist'),width=400)
        #--------------
        self.publishme()

