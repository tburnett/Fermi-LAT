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
    
    sections: load_files #EF_grades spectral_study initial_selection sed_plots pulsar_spectra
    
    remote_path: '/nfs/farm/g/glast/g/catalog/pointlike/skymodels/P8_12years/uw1208'
    local_path:  '/tmp/msp_cand/'

    query: '(dec>-57) & ( (glat>2.5) | (glat<-2.5) | (pindex<2)) '
    """
    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_files(self):
        """Load Files

        Load the csv file created using the model {self.skymodel}, at
        From `{self.model_path}`.
        (The document describing it.)[ The candidates were selected in an analysis of the unassociated sources in the model {self.skymodel}.
        The [document describing it.](https://glast-ground.slac.stanford.edu/Decorator/exp/Fermi/Decorate/groups/catalog/pointlike/skymodels/P8_12years/{self.skymodel}/plots/pulsars/index.html?skipDecoration)

        The selected sources came from an analysis of 12 years of LAT survey data which was used 
        to detect sources to be added to the 12-year 4FGL-DR3 catalog, but are probably below its threshold for inclusion.  Criteria were: 
     
        * Well=localized. Sources are localized by maximizing the likelihood as a function of position. The shape 
        of this function must be consistent with the Gaussian expectation. 
        * Not associated with a known $\gamma$-ray source.
        * No closer than 5 degrees to a 4FGL source, but the closest such cannot be a very strong source. This avoids confusion with stronger sources, 
        * Spectral shape consistent with known $\gamma$-ray pulsars. This involved limits on the correlated curvature and mean energy.   
 
        Addition selection criteria: {self.query}, result in {ncand} candidates.

        The head of the DataFrame:
        {head}
        """
        self.skymodel = getattr(self, 'version', 'uw1208')
        self.model_path = os.path.join(self.remote_path, self.skymodel)
        self.local_path = os.path.join(self.local_path, self.skymodel)
        with ftp.SLAC(self.model_path, self.local_path) as slac:
            slac.get('plots/pulsars/pulsar_candidates.csv')
        
        df =pd.read_csv(self.local_path+'plots/pulsars/pulsar_candidates.csv', index_col=0)
        self.df = df.query(self.query)
        ncand = len(self.df)
        head= self.df.head()
        #------------
        self.publishme()

class ImageTable(object):

    width=120
    row_size=8

    def __init__(self, 
                    doc,
                    image_file_path='sedfig', #'/home/burnett/work/GMRT/data/candidate_seds',
                    query= ''):



        self.df = doc.df.query(query) if query else doc.df
        names = self.df.index
        tpath = os.path.join(doc.docpath, doc.docname, image_file_path)
        for name in names:
            fn = name+'.jpg'
            shutil.copy(os.path.join(doc.tmp, 'sedfig', fn), os.path.join(tpath, fn))
            assert os.path.isfile(os.path.join(tpath, fn)), f'Failed {fn}'


        self.images = [image_file_path+f'/{name}.jpg' for name in names]
        
        #assert os.path.isfile(self.images[0])
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


