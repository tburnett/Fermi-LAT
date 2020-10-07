"""
"""

import os, sys, glob
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from utilities import SkyDir, healpix

from jupydoc import DocPublisher

__docs__= ['PSCstudy']

class PSCstudy(DocPublisher):
    """
    title: Study the 4FGL-DR2 diffuse parameters

    author: Toby Burnett
    
    sections: introduction galactic_maps plots

    diffuse_files: 
        - 'gll_iem_v07_hpx.fits'
        - 'isotropic_8years_P305_SOURCE_FRONT.txt'
    
    psc_pattern: /home/burnett/fermi/diffuse/gll_psc*.fit
    
    uw_diffuse: /home/burnett/fermi/diffuse/uw/gll_iem_uw_v3.fits

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.df = self.roi_dataframe()
        self.gflux = healpix.HPcube.from_FITS(f'/home/burnett/fermi/diffuse/{self.diffuse_files[0]}')

    def roi_dataframe(self):
        from astropy.io import fits
        from astropy.table import Table
            
        ff = sorted(glob.glob(self.psc_pattern))
        assert len(ff)>0, 'No files found with "{}"'.format(self.psc_pattern)
        self.psc_file = f = ff[-1]

        print( 'opening file "{}"'.format(f))
        df=Table.read(f, hdu='ROIS').to_pandas()
        df['singlat'] = np.sin(np.radians(df.GLAT))

        glon =df.GLON.copy()
        glon[glon>180] -=360
        df['glat']=df.GLAT
        df['glon']=glon
        df['radius'] = df.CoreRadius
        df['gal_norm']=df.PARNAM1
        df['gal_index']=df.PARNAM2
        df['iso_norm']=df.PARNAM3

        return df['glon glat radius gal_norm gal_index iso_norm '.split()]

    def introduction(self):
        """Introduction

        The point of this study of the diffuse fit parameters used by 4FGL-DR2 is to assess the validity of the galactic diffuse model. To the extent 
        that it is a valid descrition of the data (including of course a previous set of source parameters), the various corrections should be small.

        In this document I look at the 1 TeV extrapolaiton of the duffuse model, to emphasize the granularity of its patch component. There are more
        details elsewhere>

        Then I look at the corrections themselves, along two critical stripes to show how varied they are, and especially at the position of the 
        south bubble. 
        """
        #----------
        self.publishme()

    def galactic_maps(self):
        """Galactic Diffuse Model

        The diffuse model used by 4FGL is `{self.diffuse_files[0]}`. It includes a data-based measurement of the Loop I structure and Fermi bubbles using its "patch". It tends to have small-angle structure. This fact, and the limited extent is emphasized by the extrapolation to 1 TeV:
        {fig1}

        For comparison, here is the UW model:

        {fig2}

        ###A concern:
        Should we trust a model with such unrealistic structure?
        """
        fig1 = plt.figure(figsize=(12,6), num=1)
        self.gflux.ait_plot(1e6, fig=fig1 )
        fig1.caption = f'Flux from {self.diffuse_files[0]} at 1 TeV.'
        
        fig2 = plt.figure(figsize=(12,6), num=2)
        uwdiffuse = healpix.HPcube.from_FITS(self.uw_diffuse)
        uwdiffuse.ait_plot(1e6, fig=fig2 ,label='')
        fig2.caption = f'Flux from UW diffuse model at 1 TeV.'

        #---------
        self.publishme()

    def plots(self):
        r"""ROI parameter distributions

        This is a look at the ROI's and their diffuse fit parameters, found in the file `{self.psc_file}`.
        These represent a measurement, using a consistent set of point- and extended sources, of corrections to the 
        galactic diffuse model `{self.diffuse_files[0]}`.
        
        ## Positions of the {N} ROIs. 
        The color represents the size.       
        {fig1}
        The locations and sizes were adjusted to limit the number of variable sources per ROI. 

        ### Parameter values along stripes through the GC.
        The most difficult regions are along the galactic plane, the meridian that passes
        through the bubbles and Loop I. The latter features are not part of the galactic model, but were presumably accounted for by the "patch" component 
        derived using data.  

        As a reminder, here are the patch factors from 

        #### GC polar stripe
        Select the ROIs around a meridian passing through the GC. 
        
        {fig2}

        #### ROIs near the galactic plane
        {fig3}

        """
        df = self.df
        N = len(df)
        roi_sd = SkyDir.gal(df.glon, df.glat)

        def roi_map(num):
            fig = plt.figure(figsize=(10,6), num=num)
            ax = fig.add_subplot(111, projection="aitoff")
            ax.grid(color='grey')
            im=ax.scatter(-np.radians(df.glon), np.radians(df.glat.clip(-87,88)), c = df.radius, cmap='jet')
            cb = plt.colorbar(im, ax=ax, shrink=0.7, label='ROI size (deg)') 
            fig.set_facecolor('white')
            ax.set(xticklabels=[], yticklabels=[])#, xlabel='longitude', ylabel='latitude')
            fig.caption = '4FGL-DR2 ROI locations, with the color representing the ROI size.'
            return fig


        def stripe(label='Polar', num=2, energy=1000):
            # to evaluate the galactic flux
            gflux = healpix.HPcube.from_FITS(f'/home/burnett/fermi/diffuse/{self.diffuse_files[0]}')
            
            if label == 'Polar':
                cosb = np.cos( np.radians(df.glat) )
                stripe = np.abs(df.glon)<5/cosb
                blist = np.linspace(-90,90,180)
                x = df.glat[stripe]
                span = dict(xmin=-75, xmax=85, color='orange',alpha=0.1)
                xlim = (-90,90)
                xlabel = r'$b\ \mathrm{[deg]}$'
                xticks = np.linspace(-90,90,13)
                title = r'$\mathrm{{{{  {}\ Stripe: {}\ ROIs\  with}}}}\ |l|<5/ \cos(b)$'.format(
                    label,sum(stripe))
            elif label=='Planar':
                stripe = np.abs(df.glat)<2
                x = df.glon[stripe]
                y = df.glat[stripe]
                xlim = (180,-180)
                xlabel = r'$l\ \mathrm{[deg]}$'
                xticks = np.linspace(180,-180,13)
                span = dict(xmin=-90, xmax=165, color='orange',alpha=0.1)
                title = r'$\mathrm{{{{ {}\ Stripe: {}\ ROIs\  with}}}}\ |b|<2 $'.format(
                    label,sum(stripe))
            
            # iflux1000 = iso_spect(0, 1000.)
            
            fig,axx = plt.subplots(4,1, figsize=(10,12), sharex=True, num=num,
                                gridspec_kw=dict(top=0.92, left=0.15, hspace=0.05))
            
            for i,ax in enumerate(axx):
                ax.grid(alpha=0.5)
                ax.axvline(0, color='grey')
                if i>0: 
                    ax.axhline(1.0, color='grey');
                    ax.axvspan(**span)
                if i==0:# flux 
                    y = self.gflux(SkyDir.gal(df.glon[stripe], df.glat[stripe]), energy)
                    ax.semilogy(x, y, '+', label='galactic')
                    # fix this later
                    #ax.axhline(iflux1000, color='g', ls='--', label='isotropic')
                    ax.legend()
                    ax.set( ylabel=f'Flux @ {energy*1e-3:.0f} GeV')
                elif i==1: #galactic norm
                    ax.plot( x, df.gal_norm[stripe], 'o');
                    ax.set( ylabel='gal norm', ylim=(0.5,1.1));
                elif i==2: #galactic index
                    ax.plot( x, df.gal_index[stripe], 'o');
                    ax.axhline(0, color='grey')
                    ax.set( ylabel='gal index', ylim=(-0.1,0.1))#, ylim=(0.5,1.1));
                elif i==3: #iso norm
                    ax.plot(x, df.iso_norm[stripe],'o');
                    ax.set( ylabel='iso norm', ylim=(0.4, 2.05), 
                    xlim=xlim, xlabel=xlabel, xticks=xticks)
            fig.caption = title   
            fig.text( 0.03, 0.4, 'Diffuse parameters', va='center', rotation='vertical')
            return fig

        fig1 = roi_map(num=1)
        fig2 = stripe('Polar', num=2)
        fig3 = stripe('Planar',num=3)
        #--------------
        self.publishme()