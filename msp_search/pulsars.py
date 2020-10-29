"""
"""

import os, sys
import shutil
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from utilities import ftp, LogParabola

from jupydoc import DocPublisher

__docs__ = ['MSPcandidates']

class MSPcandidates(DocPublisher):
    """
    title: MSP pulsar search 
 
    author: Toby Burnett
    
    sections: introduction examine_candidates 
                further_cuts [ non_fgl_cuts fgl_cuts fgl_cuts2]
                sed_table [non_4fgl_seds fgl_seds fgl_seds2 ]

    decorator_path: https://glast-ground.slac.stanford.edu/Decorator/exp/Fermi/Decorate/groups/catalog/pointlike/skymodels/{}/plots/pulsars/index.html?skipDecoration#5
    
    """    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup_skymodel()
        
        plt.rc('font', size=16)

    def setup_skymodel(self,     
        slac_path= '/nfs/farm/g/glast/g/catalog/pointlike/skymodels/',
        local_path= '/tmp/skymodels/',
        ):

        if not self.version or self.version.split('.')[-1]=='Report':
            self.skymodel = 'uw1208-v3'
        else: self.skymodel = self.version
        mkey = self.skymodel[:4]
        
        self.year = dict(uw12='P8_12years', uw89='P305_8years', uw90='P8-10years' )[mkey]

        self.slac_path = os.path.join(slac_path, self.year,self.skymodel)
        self.local_path = os.path.join(local_path, self.skymodel)

    def check_slac_files(self, folders=['pulsars'],reload=False, quiet=False):
        with ftp.SLAC(self.slac_path, self.local_path) as slac:

            print(f'loading/checking folders to copy from SLAC:  ', end='')
            for folder in folders:
                print(f', {folder} ')
                if not os.path.isdir(os.path.join(self.local_path, folder)):
                    slac.get(f'plots/{folder}/*', reload=reload, quiet=quiet)
            #slac.get('plots/pulsars/candidates/*')
            #slac.get('plots/pulsars/pulsar_candidates_in_4fgl.csv', reload=True, quiet=False)
            # slac.get('config.yaml'); print(', config.yaml', end='')
            # slac.get('../config.yaml')
            # lp = self.local_path
            # shutil.move(os.path.join(lp, '../config.yaml'), 
            #         os.path.join(lp, 'super_config.yaml') )
            # print(', ../config.yaml -> super_config.yaml')

    def introduction(self, reload=False):
        """Introduction

        This is a continuation of a pulsar candidate selection that was performed using
        all of the information about the sky model {self.year}/{self.skymodel} at SLAC. 
        See [this output]({decorator})
        Here we copy relevant files
        to this local machine for further analysis:

        #### Files copied from SLAC:
      
        {tmpfiles}

        Loaded:
        * Fermi-LAT pulsars from file `{pfile}` with {npsr} pulsars, 
        * non-4FGL candidates from file `{cfile}` with {ncand} entries.
        * 4FGL candidates from file `{ffile}` with {ncandf} entries.
        """
        decorator = self.decorator_path.format(os.path.join(self.year, self.skymodel))
        tmp = self.local_path
        pfile = 'pulsars.csv'
        cfile = 'plots/pulsars/pulsar_candidates.csv'
        ffile = 'plots/pulsars/pulsar_candidates_in_4fgl.csv'
        with ftp.SLAC(self.slac_path, tmp) as slac:
            slac.get(pfile)
            slac.get(cfile)
            slac.get(ffile)
            
        tmpfiles = self.shell(f'ls -l {tmp}/plots/pulsars', summary='Files in the folder plots/pulsars' )

        self.dfp = pd.read_csv(tmp+'/'+pfile, index_col=0)
        self.dfc = pd.read_csv(tmp+'/'+cfile, index_col=0)
        self.dff = pd.read_csv(tmp+'/'+ffile, index_col=0)
        npsr = len(self.dfp)
        ncand = len(self.dfc)
        ncandf = len(self.dff)
        self.publishme()
    
    def examine_candidates(self):
        """Examine Pulsar Candidates


        ### The following is from the initial selection run at SLAC
        
        Make a list of sources with the selections

        * not associated
        * not in 4FGL-DR2 or withinn 0.5 deg of one
        * nearest 4FGL source is extended or has TS<1000
         
        The plots, copied from the SLAC analysis, define the sources in thses lists, showing the effects of subsequent cuts:
        * 0.15 < curvature < 0.75
        * pivot energy < 3 GeV
        * R95 < 15 arcmin
        {selection_hists}

        #### The set of sources that **were** in 4FGL.
        Other cuts are the same

        {selection_4fgl}
        """
        selection_hists = self.image(f'{self.local_path}/plots/pulsars/new_candidates_{self.skymodel}.jpg',
            width=400, caption='Selection histograms, non-4FGL')

        selection_4fgl = self.image(f'{self.local_path}/plots/pulsars/candidates_in_4fgl_{self.skymodel}.jpg',
            width=400, caption='Selection histograms, 4FGL')
        #-------
        self.publishme()

    def further_cuts(self):
        """Further cuts 

        Now we require that sources were first detected in the 10- or 12-year 4FGL updates and that the pulsar-like spectral shape had the best fit.
        This corresponds to the 4th character in the name is being "N". (The first three characters help identify the model in which they were detected.)

        These sources were fit to an exponential cutoff power-law spectral shape, for which the 
        spectral index parameter is the slope at low energies. We require that they be relatively hard, with the index less than 1.5. 

        FInally we require $|b|>2.5$, avoiding the confused galactic ridge, and biasing in favor of MSPs. (A further selection on declination would be necessary
        for a specific radio telescope.)
   
        The remaining log-parabola sources surely include pulsar candidates. The "photon index" parameter for them is the slope of the log flux vs. log energy at
        the pivot energy, typically above 2. To compare with the exponential cutoff fits, this has been extrapolated to 100 MeV in the last plot.  But for now, there 
        seem to be plenty of candidates with the pulsar-like spectral fits.

        Below are the plots for the two categories, showing the effects of the sequential selection cuts.
        """
        self.publishme()

    def selection_hists(self, 
            dfx, 
            caption='',
            tsclip=250,
            ):
        ts = dfx.ts.astype(float).clip(0,tsclip)
        singlat = np.sin(np.radians(dfx.glat.astype(float)))
        curvature= dfx.curvature.astype(float).clip(0,1)
        r95_arcmin = dfx.r95.astype(float)*60
        pivot = dfx.pivot_energy.astype(float)

        ncut = np.array([n[3]=='N' for n in dfx.index], dtype=bool)
        # photon index from fit - extrapolate to 100 MeV if log-parabola fit
        pindex = dfx.pindex.astype(float)
        adjust = pindex + curvature * np.log(100/pivot)
        pindex[~ncut] = adjust[~ncut]
        
        # the cuts to apply
        index_cut = pindex<1.5
        bcut = np.abs(singlat)>np.sin(np.radians(2.5))
        cuts = [ ncut, ncut& index_cut, ncut & index_cut & bcut ]
        cut_labels=['pulsar fit', 'index<1.5', '|b|>2.5']
        dfx.loc[:, 'keep'] = ncut & index_cut & bcut

        fig, axx = plt.subplots(2,3, figsize=(12,10))
        ax1,ax2,ax3,ax4,ax5,ax6 = axx.flatten()
        

        def doit(ax, x, bins, xlabel, xlog=False):
            hkw = dict(histtype='step', lw=2, log=False)  
            ax.hist(x, bins, label='', **hkw)  
            for i,cut in enumerate(cuts):
                if i == len(cuts)-1:
                    hkw.update(histtype='stepfilled', color='lightgray', ec='black')
                ax.hist(x[cut], bins, label=cut_labels[i], **hkw)
            ax.set(xlabel=xlabel, xscale='log' if xlog else 'linear')
            #ax.set(ylim=(0.9,None));
            ax.legend(prop=dict(size=10))
      

        doit(ax1, curvature, np.linspace(0,1,21), 'curvature')
        doit(ax2, pivot, np.logspace(np.log10(200),np.log10(2e4),21), 'pivot energy', xlog=True)
        doit(ax3, r95_arcmin, np.linspace(0,25,26),'R95 (arcmin)')
        doit(ax4, ts, np.logspace(1,np.log10(tsclip),25), 'TS', xlog=True)
        doit(ax5, singlat, np.linspace(-1,1,21), 'sin(b)')
        doit(ax6, pindex.clip(0.,3), np.linspace(0,3,31), 'low-energy index')
        if not caption:
            fig.caption='Same plots as the SLAC selection, but with new cuts on source detection properties.'\
                        ' The last plot is the low-energy photon index, see text.'
        fig.width = 600
        fig.set_facecolor('white')
        return fig

    def non_fgl_cuts(self):
        """Non-4FGL sources

        {fig}
        """
        fig = self.selection_hists(self.dfc)
       
        #------
        self.publishme()
    
    def fgl_cuts(self):
        """4FGL sources

        {fig}
        """
        fig = self.selection_hists(self.dff)

        #------------
        self.publishme()

    def fgl_cuts2(self):
        """4FGL sources, part 2

        Here we examine 4FGL-DR2 candidate sources that did not originate from seeds detected as likely pulsars.
        
        For these, we have the log parabola parameters, this section is devoted to understanding how to 
        interpret them to select the most likely pulsars
        An example is {example}, or {sname}, an obvious-looking example for which pulsations have
        been looked for, but not found.
        {fig1}

        Here are the distributions of the spectral parameters, with the pivot flux, and the spectral index evaluated at 100 MeV.


        {fig2}


        """
        # select sources with "N" not 4th caracter of name
        k = self.dff.apply(lambda x: x.name[3]!='N', axis=1).array
        dff = self.dff[k].copy()
        print(f'Selected {sum(k)} SLAC sources not tagged as pulsar-like')

        # extract the parameters from the DataFrame--have to evaluate the string "pars" first
        dff.loc[:,'p'] = dff.pars.apply(lambda x: eval(x.replace(' ',',')))
        dff.loc[:, 'logpar' ] = dff.p.apply(lambda p: LogParabola(p)) 
        dff.loc[:, 'f100'] = dff.logpar.apply(lambda f: f(100))
        dff.loc[:, 'a100'] = dff.logpar.apply(lambda f: f.alpha(100))
        pivot = dff.p.apply(lambda p:p[3])
        dff.loc[:, 'peak'] = dff.logpar.apply(lambda f: f.peak)
        dff.loc[:,'flux'] = dff.apply(lambda row: row.logpar(row.peak), axis=1)
        dff.loc[:,'a20k'] = dff.logpar.apply(lambda f: f.alpha(2e4) )

        cut =   (dff.f100>0.01)  &  \
                (dff.a100<1.5) & \
                (np.abs(dff.glat)>2.5) & \
                (dff.a20k>3)

        print(f'Cut selects {sum(cut)}')
        self.dffk = dff[cut]  

        ### the example source
        example = 'P88Y2837'

        f, sname, ts = dff.loc[example, ['logpar', 'sname', 'ts']]
        fig1 = f.sed(sname)
        fig1.caption=f'SED for source {example}, or {sname}. It has TS = {ts:.0f}'

        fig2,  axx = plt.subplots(2,3, figsize=( 15,10), num=2)
        fig2.width = 600
        fig2.caption = r'Spectral parameters and $\sin(b)$, showing effect of selection cut.'
        ax1, ax2, ax3, ax4, ax5, ax6 = axx.flatten()

        hkw= dict(histtype='step', lw=2)
        hkw2 = {**hkw, **dict(histtype='stepfilled',color='lightgray', ec='black')}

        def dohist(ax, x, bins, xlabel='', xscale='linear'):
            ax.hist(x, bins, **hkw)
            ax.hist(x[cut], bins,  **hkw2)
            ax.set(xscale=xscale, xlabel=xlabel, ) 
            ax.grid(alpha=0.5) 

        dohist(ax1, dff.flux, np.logspace(-1.5,1.5,31), xscale='log', xlabel='Flux @ peak')
        dohist(ax2, dff.a100.clip(0,3), np.linspace(0,3,31), xlabel='Index @100 MeV')
       
        ax3.plot(dff.flux.clip(1e-2,1e2), dff.a100.clip(0,3), '.')
        ax3.set(xscale='log', ylim=(0,3.05), xlabel='Flux @ peak', ylabel='100 MeV index')
        a,b = dff.loc[example, 'flux a100'.split()]
        ax3.plot(a,b, 'or', label=f'{example}')
        ax3.legend()
        ax3.grid(alpha=0.5) 
       
        curvature = dff.p.apply(lambda p: p[2])
        dohist(ax4, curvature.clip(0,2), np.linspace(0,1, 26), xlabel='curvature')

        dohist (ax5, dff.peak, np.logspace(2,4,21), xscale='log', xlabel='peak energy')
        
        singlat = np.sin(np.radians(dff.glat))
        dohist(ax6, singlat, np.linspace(-1,1,41), xlabel=r'$\sin(b)$')
        #--------------
        self.publishme()

    def sed_table(self):
        """Tables of SEDs
        
        For each data set, here are links to csv files containing essential information, and tables of SED images.
        """
        self.publishme()

    def non_4fgl_seds(self):
        """Selected non-4FGL seds

        A table of the {N} non-4FGL pulsar candidates, sorted with decending TS:

        <a href="pulsar_candidates.csv">Candidate csv table</a>
        (Note that Chrome has a bug: It may change the extension to "xls".)
        
        {df}
        

        And simple SED's, all with same scale 100 MeV-30 GeV and 0.6 - 40 $\mathrm{{ eV (s\ cm^2 )^{-1} }}$. 
        {images}
        """
        df = self.dfc.query('keep==True').copy()
        df.loc[:,'name'] = df.index
        df = df['name ra dec ts glat r95'.split()]
        df.sort_values(by='ts', ascending=False)
        df.to_csv(os.path.join(self.doc_folders[0], 'pulsar_candidates.csv') )
        N = len(df)
        images = ImageTable(self,
            os.path.join(self.local_path, 'plots/pulsars/candidates'),
            df)
        #------
        self.publishme()
    
    def fgl_seds(self):
        """Selected 4FGL seds

        A table of the {N} 4FGL pulsar candidates sorted with decending TS:

        <a href="pulsar_candidates_4fgl.csv">Candidate csv table</a>
        (Note that Chrome has a bug: It may change the extension to "xls".)
        
        {df}        

        And simple SED's, all with same scales, 100 MeV-30 GeV and 0.6 - 40 $\mathrm{{ eV (s\ cm^2 )^{-1} }}$. 
        {images}
        """
        df = self.dff.query('keep==True').copy()
        df.loc[:,'name'] = df.index
        df = df['name ra dec ts glat r95 sname'.split()]
        df.sort_values(by='ts', ascending=False)
        df.to_csv(os.path.join(self.doc_folders[0], 'pulsar_candidates_4fgl.csv') )
        N = len(df)
        images = ImageTable(self,
            os.path.join(self.local_path, 'plots/pulsars/candidates'),
            df)
        #------
        self.publishme()

    def fgl_seds2(self):
        """4FGL seds, part 2

        These were selected above.

        Showing {n} out of {nt} sorted by TS, 

        {images}
        """
        n = 100; nt=len(self.dffk)
        images = ImageTable(self, os.path.join(self.local_path, 'plots/pulsars/candidates'), self.dffk[:n]) 

        self.publishme()



class ImageTable(object):

    width=120
    row_size=8

    def __init__(self,
                    doc: 'the document class', 
                    source_path:'where to find the original images',
                    df:'data frame', #names:'list of source names',
                    image_file_path='images',
                ):

        # copy each image from the source path to both local and document
        for name in df.index:
            fn = name+'.jpg'
            a = os.path.join(source_path, fn)
            for folder in doc.doc_folders:
                b = os.path.join(folder, image_file_path, fn)
                shutil.copy(a, b)
        self.images = [image_file_path+f'/{name}.jpg' for name in df.index]
        self.ts = df.ts
                
        print(f'Will display {len(self.images)} images') 

    def _repr_html_(self):
        # this allows display in a cell 

        def image_rep(image, ts):
            _, fname = os.path.split(image)
            name, _ = os.path.splitext(fname)
            style='width: 120px; margin: 0px; float: left; border: 1px solid black;'
            return f'<a href="{image}"><img src="{image}" styple="{style}"'\
                   f' width={self.width} alt="file {image}" title="{name}, {ts:.0f}"/></a>'

        imgs = self.images
        rs = self.row_size
        rows = (len(imgs)+rs-1)//rs

        ret = '\n<table>'
        j=0
        for row in range(rows):
            ret += '\n  <tr>'
            for i in range(self.row_size):
                ret +=  f'\n    <td class="td"> {image_rep(imgs[j], self.ts[j] )}</td>'
                j +=1
                if j==len(imgs): break 
            ret += '\n   </tr>'
        ret += '\n</table>'
        
        return ret

    def __str__(self):
        return self._repr_html_()
