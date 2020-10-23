"""

"""
import os, sys
import shutil, pickle
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib
from astropy.io import fits

from utilities import SkyDir, HPmap, HPcube, ait_plot, ait_multiplot
from utilities import ftp, healpix

from jupydoc import DocPublisher

__docs__ = ['MiscPlots']

class MiscPlots(DocPublisher):
    """
    title: Various plots

 
    local_fermi_path: '/home/burnett/fermi'

    dm7: 'gll_iem_v07_hpx.fits'
    nopatch_diffuse_file: 'test_model_InnerGalaxyYB01_test512_interp_nopatch.fits'
    nopatch_noCO9:        'test_model_InnerGalaxyYB01_test512_interp_noCO9_lg_nopatch.fits'

    """

    def title_page(self):
        """

        <h2> Miscellaneous all-sky plots</h2>

        {fig1}
        {fig2}
        {fig3}
        {fig4}
        """
        plt.rc('font', size=16)

        def flux_plot(model_file=self.dm7, caption='', energy=1000.,):
            dm= HPcube.from_FITS(os.path.join(self.local_fermi_path,'diffuse', model_file))

            fig, ax = plt.subplots(figsize=(20,10), subplot_kw=dict(projection='aitoff') , num=self.newfignum() )
            dm.ait_plot(energy, pixelsize=0.25, ax=ax, cb_kw=dict(shrink=0.7), label=f'{energy:.0f} MeV' )
            fig.caption= caption
            fig.width=800
            return fig
        fig1 = flux_plot(self.dm7, caption=f'Released model, with the patch and "noCO9" {self.dm7}')
        fig2 = flux_plot(self.nopatch_diffuse_file, caption='The UW no-patch starting point, indludes C09')
        fig3 = flux_plot(self.nopatch_noCO9, caption='No-patch and "noCO9"')
        fig4 = flux_plot('uw/gll_iem_uw_v4.fits', caption='UW diffuse model v4')

        # do the ratio like Gulli?
        
        
        self.publishme()
