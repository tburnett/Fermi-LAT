"""
"""

import os, sys
import shutil
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from utilities import ftp

from jupydoc import DocPublisher

__docs__ = ['Pulsars']

class Pulsars(DocPublisher):
    """
    title: MSP pulsar search
 
    author: Toby Burnett
    
    sections: load_file get_candidates further_cuts sed_table

    #query: '(dec>-57) & ( (glat>2.5) | (glat<-2.5) | (pindex<2)) '
    """
    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup_skymodel()

    def setup_skymodel(self,     
        slac_path= '/nfs/farm/g/glast/g/catalog/pointlike/skymodels/',
        local_path= '/tmp/skymodels/'):

        if not self.version or self.version.split('.')[-1]=='Report':
            self.skymodel = 'uw1208-v3'
        else: self.skymodel = self.version
        mkey = self.skymodel[:4]
        
        year = dict(uw12='P8_12years', uw89='P305_8years', uw90='P8-10years' )[mkey]

        self.slac_path = os.path.join(slac_path, year,self.skymodel)
        self.local_path = os.path.join(local_path, self.skymodel)

    def load_file(self, reload=False):
        """Set up files

        #### Files copied from SLAC to {tmp};
      
        {tmpfiles}

        Loaded:
        * Fermi-LAT pulsars from file `{pfile}` with {npsr} pulsars, 
        * candidates from file `{cfile}` with {ncand} entries.
        """
        tmp = self.local_path
        pfile = 'pulsars.csv'
        cfile = 'plots/pulsars/pulsar_candidates.csv'
        with ftp.SLAC(self.slac_path, tmp) as slac:
            slac.get(pfile)
            slac.get(cfile)
        tmpfiles = self.shell(f'ls -l {tmp}')

        self.dfp = pd.read_csv(tmp+'/'+pfile, index_col=0)
        self.dfc = pd.read_csv(tmp+'/'+cfile, index_col=0)
        npsr = len(self.dfp)
        ncand = len(self.dfc)
        self.publishme()
    
    def get_candidates(self):
        """Examine Pulsar Candidates


        ### The following is from the initial selection run
        
        Make a list of sources with the selections

        * not associated
        * not in 4FGL or withinn 0.5 deg of one
        * nearest 4FGL source is extended or has TS<1000
         
        The plots are for sources from this list, showing the effects of subsequent cuts:
        * 0.15 < curvature < 0.75
        * pivot energy < 3 GeV
        * R95 < 15 arcmin
        {selection_hists}
        """
        selection_hists = self.image(f'{self.local_path}/plots/pulsars/new_candidates_{self.skymodel}.jpg',
            width=600, caption='Selection histograms')
        #-------
        self.publishme()

    def further_cuts(self):
        """Further cuts

        Now require that when the source was detected, the pulsar-like spectral shape had the best fit (the 4th character in the name is "N").

        These sources were fit to an exponential cutoff power-law spectral shape, for which the 
        spectral index parameter is the slope at low energies. Make the relative hard 
        with a further cut of 1.5.

        (The index for the log parabola fits is the slope at the pivot energy) 
        
        {fig}
        """

        dfx =self.dfc
        ts = dfx.ts.astype(float).clip(0,1000)
        singlat = np.sin(np.radians(dfx.glat.astype(float)))
        curvature= dfx.curvature.astype(float).clip(0,1)
        r95_arcmin = dfx.r95.astype(float)*60
        pivot = dfx.pivot_energy.astype(float)
        #eflux =dfx.eflux.astype(float)
        ncut = np.array([n[3]=='N' for n in dfx.index], dtype=bool)
        index_cut = dfx.pindex<1.5
        cuts = [ ncut, ncut& index_cut ]
        cut_labels=['pulsar fit', 'index<1.5']
        self.keep = ncut & index_cut

        plt.rc('font', size=16)

        fig, axx = plt.subplots(2,3, figsize=(12,10))
        ax1,ax2,ax3,ax4,ax5,ax6 = axx.flatten()
        hkw = dict(histtype='step', lw=2, log=True)    
        def doit(ax, x, bins, xlabel, xlog=False):
            ax.hist(x, bins, label='', **hkw)  
            for i,cut in enumerate(cuts):
                ax.hist(x[cut], bins, label=cut_labels[i], **hkw)
            ax.set(xlabel=xlabel, xscale='log' if xlog else 'linear')
            ax.set(ylim=(0.9,None));
            ax.legend()

        doit(ax1, curvature, np.linspace(0,1,21), 'curvature')
        doit(ax2, pivot, np.logspace(np.log10(200),np.log10(2e4),21), 'pivot energy', xlog=True)
        doit(ax3, r95_arcmin, np.linspace(0,25,26),'R95 (arcmin)')
        doit(ax4, ts, np.logspace(1,2.4,25), 'TS', xlog=True)
        doit(ax5, singlat, np.linspace(-1,1,21), 'sin(b)')
        doit(ax6, dfx.pindex, np.linspace(1,3,21), 'photon index')
        fig.caption='Same plots as selection, but with new cut on source detection type'
        fig.width = 600
        
        #------
        self.publishme()

    def sed_table(self):
        """Table of SEDs
        
        {images}
        """


        names = sorted(self.dfc[self.keep].index)   
        images = ImageTable(self,
            os.path.join(self.local_path, 'plots/pulsars/candidates'),
            names, )
        #------
        self.publishme()


class ImageTable(object):

    width=120
    row_size=8

    def __init__(self,
                    doc: 'the document class', 
                    source_path:'where to find the original images',
                    names:'list of source names',
                    image_file_path='images',
                ):

        # copy each image from the source path to both local and document
        for name in names:
            fn = name+'.jpg'
            a = os.path.join(source_path, fn)
            for folder in doc.doc_folders:
                b = os.path.join(folder, image_file_path, fn)
                shutil.copy(a, b)
        self.images = [image_file_path+f'/{name}.jpg' for name in names]
        
        print(f'Will display {len(self.images)} images') 

    def _repr_html_(self):

        def image_rep(image):
            _, fname = os.path.split(image)
            name, _ = os.path.splitext(fname)
            style='width: 120px; margin: 0px; float: left; border: 1px solid black;'
            return f'<img src="{image}" styple="{style}" width={self.width} alt="file {image}" title={name} />'

        imgs = self.images
        rows = len(imgs)//self.row_size

        ret = '\n<table>'
        j=0
        for row in range(rows):
            ret += '\n  <tr>'
            for i in range(self.row_size):
                ret +=  f'\n    <td class="td"> {image_rep(imgs[j])}</td>'
                j +=1
            ret += '\n   </tr>'
        ret += '\n</table>'
        
        return ret

    def __str__(self):
        return self._repr_html_()
