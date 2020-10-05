"""

"""
import os, sys
import shutil, pickle
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from astropy.io import fits

from utilities import SkyDir, HPmap, HPcube, ait_plot, ait_multiplot, Polyfit
from utilities import ftp, healpix

from jupydoc import DocPublisher

__docs__ = ['UWdiffuseModel','CheckRatio', 'ResidualAnalysis']

bubble_path = '/home/burnett/fermi/bubbles/'


class CheckRatio(DocPublisher):
    """
    title: Study the diffuse count ratios\n
         Version {version}
    sections: get_counts [counts_v2] cube_comparison cubic_fit [cubic_fit_v2]

    slac_path:  '/afs/slac/g/glast/groups/catalog/pointlike/fermi/skymodels/P305_8years'
    slac_fermi_path: '/afs/slac/g/glast/groups/catalog/pointlike/fermi'
    local_fermi_path: '/home/burnett/fermi'
    local_path: '/tmp/uw_diffuse/roi_info'

    """
    def __init__(self, **kwargs):
        # self.client_mode = kwargs.pop('client_mode', False)
        # self.version = kwargs.pop('version', '')
        super().__init__(**kwargs)
        self.setup()

    def setup(self):
        self.band_energies = np.logspace(2.125, 3.875, 8)
        self.band_labels = list(map(lambda x: f'{x/1e3:.2f} GeV', self.band_energies))
        self.sigma_v2=20.
        if self.client_mode:
            self.version='v2'
        else:
            assert self.version=='v2', f'only support version v2, found {self.version}'

    def get_counts(self, nside=64, sigma=2.5):
        r"""Get the counts from SLAC processing

        Processing of each model with the UW pipeline saves the predicted counts from each diffuse source for each ROI.
        From this I save a file `diffuse_counts.pkl` with the eight energy bands from 100 MeV to 10 GeV, combining Front
        and Back where appropriate. This files are downloaded here and processed.

        Three models are analyzed:
        * uw8607: the 4FGL progenitor, and the "UW gold standard" ad-hoc galactic and isotropic files. Call this "v0".
        * uw8607g: same as uw8607 but evaluated with the Fermi-standard gll_iem_v07 galactic model.
        * uw8607-np: A fairly recent diffuse model that does not include the "patch" used in its fitting. 
        * uw8607-v2: Application of the version v2 = see subsection below

        It is `test_model_InnerGalaxyYB01_test512_interp_nopatch.fits`

        The idea is to determine adjustments to the nopatch model so that the predicted counts will coincide. 

        
        These maps were made with nside=12, corrsponding to the ROIs, but were resampled to nside={nside}
         and smoothed with a sigma of ${sigma}^\circ$.

        ## UW - nopatch ratio
        This should be close the the factor cube used to derive the UW model
        {fig1}
        
        ## UW - gal_v07 ratio

        Here we see the differences between the UW and the gal_v07 models.

        {fig2}

        Notes:
        *  the sharp boudaries on the meridias $l=80$ and $l=-90$, the range of the diffuse model patch component.
        *  Large deviations at the position of the source bubble.
        *  Spots along the glactic plane
        *  Dramatic spot at higher energies at $l=30, b=80$
        """
        models = [f'uw8607{mdl}' for mdl in ['', 'g', '-np', '-v2']]
        # get the files frmom SLAC or the cache
        with ftp.SLAC(self.slac_path, self.local_path) as slac:
            #print( slac.listdir())
            slac.get([f'{mdl}/diffuse_counts.pkl' for mdl in models ], reload=False)   
                            
        self.diffuse_counts = dc = []
        for mdl in models:
            with open(f'{self.local_path}/{mdl}/diffuse_counts.pkl', 'rb') as imp:
                dc.append(  pickle.load(imp, encoding='latin1'))

        gal_uw = dc[0]['gal']
        gal_v07 = dc[1]['gal']
        nopatch = dc[2]['gal']
     

        # cut on b for iso check:
        b = SkyDir.from_healpix(list(range(12**3)),nside=12).galactic[1]      
        hilat = np.abs(b)>5

        #### in case I need this ####
        # self.iso_uw = dc[0]['iso']
        
        # b = SkyDir.from_healpix(list(range(12**3)),nside=12).galactic[1]      
        # hilat = np.abs(b)>5
        # ihi = np.arange(len(hilat))[hilat]
        # iso_hilat = iso[ihi,:]
        # iso_hilat.mean(axis=0), iso_hilat.std(axis=0)
       
        energies = self.band_energies
        rc12 = HPcube(gal_uw/nopatch,energies)
        
        # resample to new nside, and smooth using HPcube constructur
        self.sd_list = sd = SkyDir.from_healpix(range(12*nside**2), nside=nside)
        rc64 = HPcube(rc12(sd,energies), energies, sigma=sigma)
        self.ratio_cube = rc64 
        # fig1 =  healpix.ait_multiplot( self.ratio_cube, energies, labels=self.band_labels, 
        #     fignum=1, title=rf'Ratio of gal_uw to nopatch')

        fig1 =  healpix.ait_multiplot( self.ratio_cube, energies, 
            labels=self.band_labels,
            nx=4, cb_shrink=0.6,sizey=4, 
            fignum=1, title=rf'Ratio of gal_uw to nopatch')


        # same with this ratio
        rc12x = HPcube(gal_v07/gal_uw,energies)
        rc64x = HPcube(rc12x(sd,energies), energies, sigma=sigma)

        fig2 =  healpix.ait_multiplot( rc64x, energies, labels=self.band_labels,
            nx=4, cb_shrink=0.6,sizey=4, 
            vmin=0.75, vmax=1.25,
            fignum=2, title=rf'Ratio of gal_v07 to gal_uw')

        #----------------
        self.publishme()

    def counts_v2(self):
        """Counts from v2

        Compare the application of version v2 with v0, the "gold standard" uw8607 diffuse model.

        Since this does not show small angular scale variations, except for a few 
        ROIs that are distorted by bad fits (and the LMC), I smooth by sigma ${self.sigma_v2}^\circ$.

        {fig4}

        
        """
        models = ['-v2']
        with ftp.SLAC(self.slac_path, self.local_path) as slac:
            #print( slac.listdir())
            slac.get([f'uw8607{mdl}/diffuse_counts.pkl' for mdl in models ], reload=True)   

        dc=self.diffuse_counts
        sd = self.sd_list 
        energies = self.band_energies

        ratio = dc[3]['gal']/dc[0]['gal']
        rcube12 = HPcube( ratio, energies) 
        rcube64 = HPcube(rcube12(sd,energies), energies, sigma=self.sigma_v2)
        self.ratio_cube_v2=rcube64 
        
        fig4 =  healpix.ait_multiplot(rcube64, energies, labels=self.band_labels,
            nx=4, cb_shrink=0.6,sizey=4,  
            fignum=4, title=rf'Count ratios of v2/v0', vmin=0.9, vmax=1.1)
        #---------------
        self.publishme()

    def cubic_fit_v2(self):
        """Cubic fit for v2

        This is the log-cubic representation derived from versions v2/v0  spectral ratio:
        {fig5}

        And also the spectrum and fit at $(0,-50)$:
        {fig6}

        Finally, make a new fit to the product :

        {fig7} 
        with again that point: 
        {fig8}

        """
        cubefitv2 = healpix.Polyfit(self.ratio_cube_v2)
                    
        self.ratio_cubicfit_v2 = cubefitv2
        fig5 = cubefitv2.ait_plots(fignum=5);
        
        fig6, ax6 = plt.subplots(figsize=(4,3), num=6)
        cubefitv2.plot_fit(0,-50, ax=ax6)

        self.ratio_cubicfit = healpix.Polyfit.from_product(
                        self.ratio_cubicfit,cubefitv2)

        fig7 = self.ratio_cubicfit.ait_plots( fignum=7)
        fig8, ax8 = plt.subplots(figsize=(4,3), num=8)
        self.ratio_cubicfit.plot_fit(0,-50, ax=ax8)
        #--------
        self.publishme()

    def cube_comparison(self):
        """Cube comparison

        How does the ratio derived from the count predection maps compare with my earlier factor cube?

        The ratio of the bubble cube to the count-ratio
        {fig3}

        Clearly there is an odd problem, prompting me so recreate the factor cube.

        """ 
        # get or verify the bubbbles files
        with ftp.SLAC(self.slac_fermi_path, self.local_fermi_path) as slac:
            slac.get('diffuse/bubbles/*')

        bubble_cube = HPcube.from_FITS(os.path.join(self.local_fermi_path,'diffuse/bubbles/bubble_cube_v4.fits'))

        bc_spectra =  bubble_cube.spectral_cube  
        ratio_spectra = self.ratio_cube.spectral_cube

        r2 = HPcube(bc_spectra/ratio_spectra, self.band_energies)
        fig3 = healpix.ait_multiplot(r2, self.band_energies, 
            nx=4, cb_shrink=0.6,sizey=4, 
            labels=self.band_labels, vmin=0.9, vmax=1.1, fignum=3)
        #healpix.ait_multiplot(bubble_cube, self.band_energies, title='bubble cube')

        #-----------
        self.publishme()

    def cubic_fit(self):
        r"""Log-Cubic fit to the ratio cube

        Given the curious artifacts in the ring at $60^\circ$ I will run the cubic analysis on the gal_uw to nopatch ratio spectra.

        #### The dramatic south bubble at $(0,-50)$ is as before:
        {fig4}

        #### The cubic fit parameters (in log space)
        {fig5}
        Note the residual map: most of the sky is fit quite well, with a slight problem 
        near the south bubble, and something worth looking at near the equatorial poles.
        """

        cubefit = healpix.Polyfit(self.ratio_cube)
                    
        fig4, ax4 = plt.subplots(figsize=(4,3), num=4)
        cubefit.plot_fit(0,-50, ax=ax4)

        fig5 = cubefit.ait_plots(fignum=5);
        self.ratio_cubicfit = cubefit

        #===========
        self.publishme()


class UWdiffuseModel(DocPublisher):
    """
    title:  Release of the UW-derived Diffuse Model \n
         Version {version}

    author: Toby Burnett


    sections:   introduction galactic [flux_factor_cube apply_factors ] isotropic summary

    fermi_path: '/home/burnett/fermi'
    nopatch_diffuse_file: 'diffuse/test_model_InnerGalaxyYB01_test512_interp_nopatch.fits'
    factor_file: 'bubbles/bubble_cube_v4.fits'
    test_model_name: 'YB01_nopatch'
    diffuse_model_name: 'gll_iem_v07'
    diffuse_v07: 'diffuse/gll_iem_v07_hpx.fits'
    iso_path:  '/home/burnett/fermi/diffuse/isotropic_uw/'
    outfiles: ['gll_iem_uw_*.fits',  'gll_iso_uw_*.txt' ]
    slac_uw_path: '/nfs/farm/g/glast/g/catalog/pointlike/fermi/diffuse/uw'
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.version = getattr(self,'version', 'x')
        assert self.version=='v3', 'only support version v3'
        self.outfiles = list(map(lambda f:f.replace('*', self.version), self.outfiles))

    def introduction(self):
        """Introduction

        In the process of deriving a UW pointlike-based sky model for 4FGL source detection and localization,
        I developed an alternate technique to add a data-based component to the galactic diffuse model. I explained this in detail in [the document](https://confluence.slac.stanford.edu/display/~burnett/4FGL+source+list+announcement). The resulting map cube had no need for further ROI-based corrections. 
        The generated, and truly isotropic diffuse part, was discussed there as well. The result was that this skymodel was really all-sky, with no ROI dependence. My residuals were were much better than any previous approach.

        I had hoped at the time, 2 years ago, that this would generate some interest in exploring it, or even adopting something like it, for gtlike applications. However, everyone who
        had worked for around a year was exhausted, and there was no apparent need to revise the 
        standard procedure, which continues to require adjustment of both galactic and isotropic source fluxes for each ROI. Also, I had not created files suitable for use by gtlike.

        Now, with the advent of DR3, I've decided to pick up this torch, by at least generating the suitable files for comparison, and adopting them directly for the UW sky model generation.

        """
        #-------
        self.publishme()

    
    def galactic(self):
        """Galactic Component

        This reviews and updates the creation of the "factor cube" correction to the `gardian`-generated maps.

        The Galactic diffuse model `{self.diffuse_model_name}` (DM7) used for the 4FGL catalog is described in its paper.

        This section describes the application of a "factor cube" to the special galactic diffuse model {self.test_model_name}, 
        defined by the file `{self.nopatch_diffuse_file}`.

        {fig1}

        """
        self.dmx = dm= HPcube.from_FITS(os.path.join(self.fermi_path, self.nopatch_diffuse_file))
        energies = np.logspace(2, 5, 4)
        labels = ['0.1 GeV', '1 GeV','10 GeV','100 GeV'] #list(map(lambda x: f'{x/1e3:.1f} GeV', energies))
        fig1 = ait_multiplot(dm, energies, labels=labels, log=True, fignum=1)
        fig1.caption = f'Flux for the "patchless" Galactic diffuse test model "{self.test_model_name}"'
     
        #---------------
        self.publishme()

    def flux_factor_cube(self):
        """The Flux Factor Cube
        
        I described the procedure to generate a factor cube in this 
        [section of my 4FGL announcement](https://confluence.slac.stanford.edu/display/~burnett/4FGL+source+list+announcement#id-4FGLsourcelistannouncement-TheXC02_v3model). A subsequent more detailed examination of the properties of quadratic fits to the eight planes is [here](https://confluence.slac.stanford.edu/display/~burnett/2018/11/30/A+more+detailed+look+at+the+XC02+flux+factor+maps).
        
        That used the earlier test XC02_v3 version. I'll assume that the final one, {self.test_model_name}, would have similar corrections, but more reliable galactic features.

        My procedure generated eight maps for the energy bands from 100 MeV to 10 GeV.        

        That cube did not work so well, so I adopted a "reverse-engineering" approach, described in 
        <a href={link}>this document.</a>
        The cube generated from the ratio of the "UW best" model and the "no patch" described above was fit to a log-cubic there,
        with these parameters:

        {fig4}

        """

        print('calling CheckRatio.v2 ...')
        otherdoc, link = self.docman.client('CheckRatio.v2')
        #otherdoc = CheckRatio(client_mode=True, version='v2')
        #otherdoc()
        #link = 'no link'

        self.pf = otherdoc.ratio_cubicfit
        fig4 = self.pf.ait_plots( fignum=4)

        fig4.caption = f'Log-cubic parameters from count ratios.'
        #---------------
        self.publishme()

    def dm7_comparison(self):
        """Comparison with the DM7 model

        This "factor cube" is equivalent to the "patch" component of the diffuse model, except of
        course the latter does not cover the whole sky, and cannot account for an overprediction 
        since it is constrained to represent a positive flux. After the fact, I can derive the
        current equivalent by dividing the released model DM7 by the test model {self.test_model_name}.
        This is not exactly the patch for the released model since there is not a "no-patch" version of it&mdash;
        this is responsible for the high-latitude spots.)

        Here it is at 1 GeV and 1 TeV, with the portions below 1.1 blanked out. 
        {fig1}
        Note that there are spots at 1 TeV where the patch exceeds a ratio of 50.
        """

        cube7 = self.dm7.spectral_cube
        cubex = self.dmx.spectral_cube
        r[z]=np.nan
        ratio_cube = HPcube(r, self.dm7.energies)
        fig1.caption=f'Ratio of the diffuse model DM7 to the test model {self.test_model_name},'\
            ' at 1 GeV and 1 Tev, showing portions above 1.1'

        #---------------
        self.publishme()

    def apply_factors(self):
        """Application of the factor cube

        I derive an equivalent diffuse model by simply multiply the flux by the factor. This is equivalent to the way the pointlike all-sky system treated the two factors, so should give equivalent results. 

        The class which performs the quadratic fit to the individual planes also implements what I call a "sky function". 
        That is, it defines a member function `__call__(coord, energy)` where:
         * `coord` is an object of my class `SkyDir`. It wraps an instance of an `astropy.coords.SkyCoord`, but is more convenient. A big difference with the fermipy version is that is supports multiple coordinate values, and converstion to/from HEALPix.
         * `energy` is one or more energies in MeV. 

         Both the class `HPcube` that implements a HEALPix spectral cube, and the factor cube described in the last 
         section implement this 
         interface. All-sky plots are easily generated by passing one of these objects, and an energy, 
         to my `ait_plot`.

        This code shows how I multiply the DM7 model by the factor cube, representing the result with the 
        DM7 set of 28 energy planes, but nside={nside}.        
        The code to do this illustrates the power of this interface. In the example below `a` and  `b` are the two
         `HPcube` objects representing the initial diffuse model and the factor cube
        
        ```
        nside=256
        skygrid = SkyDir.from_healpix(range(12*nside**2), nside=nside)
        scube=np.vstack( [a(skygrid, energy) * b(skygrid, energy) for energy in a.energies] )
        dmx = HPcube(scube, a.energies, a.unit) 
        ```

        The result is written to the local file `{outfile}` and copied to the SLAC folder {self.slac_uw_path}.

        ```        
       
        dmx.to_FITS(outfile)
        ```

        {lsfile}

        It was subsequently copied to SLAC at `{self.slac_uw_path}`
        
        """
        a, b = self.dmx, self.pf
        energies, unit = a.energies, a.unit
        nside=256

        skygrid = SkyDir.from_healpix(range(12*nside**2), nside=nside)
        scube=np.vstack( [a(skygrid, energy)*b(skygrid, energy) for energy in energies])
        self.dmuw = HPcube(scube, energies, unit) 

        outfile = self.outfiles[0]
        self.dmuw.to_FITS(outfile)
        lsfile = self.shell(f'ls -l {outfile}')

        with ftp.SLAC(self.slac_uw_path, '.' ) as slac:
            slac.put(outfile, outfile)
        #---------------
        self.publishme()

    def isotropic(self):
        """Isotropic Component

        The 4FGL-uw announcement has a [section](https://confluence.slac.stanford.edu/pages/viewpage.action?spaceKey=~burnett&title=4FGL+source+list+announcement#id-4FGLsourcelistannouncement-Isotropicfits) showing the Front and Back spectra 
        which were used, and the measured scale factors applied to my eight energy bins. Note that the Front and Back values
        are very close. Since there is no longer background source that differentiates the two, (especially if the poles are excluded, or as in my case, energies below 300 MeV entirely excluded for Back), I will henceforth neglect the difference.

        A subsequent update of the procedure actually used the files `isotropic_8years_P305_SOURCE_*_XC04_interp2.txt`, where
        "*" represents FRONT or BACK. Its values, and the derived rescaling, where similar. 
        
        {isoflux_image}

        This file is saved as `{isofile}`.
        """

        isoflux_image = self.image(self.iso_path+'isotropic_spectrum.png', width=600,
                    caption='Isotropic spectra, showing the adustment', )
        isofile = self.outfiles[1]

        #-------
        self.publishme()

    def summary(self):
        """Summary

        The version {self.version} files can be found at the SLAC path {self.slac_uw_path}, filenames

        `{self.outfiles}`.

        These need to be checked, and perhaps iterated with the 12-year data set.  
        """
         #-------
        self.publishme()


class ResidualAnalysis(DocPublisher):
    """
    title: Analyis of residuals and galactic diffuse norms

    sections: get_file normalization_maps residual_maps

    slac_path: '/nfs/farm/g/glast/g/catalog/pointlike/skymodels/'
    local_path: '/tmp/skymodels/'
    """

    def get_file(self):
        """Load the roi info file 

        Source path: {self.slac_path}
        
        <br>Local path contents:
        {dir_info}

        """
        self.skymodel = self.version or  'uw1208-v3'
        mkey = self.skymodel[:4]

        year = dict(uw12='P8_12years', uw89='P305_8years' )[mkey]

        self.slac_path = os.path.join(self.slac_path, year,self.skymodel)
        self.local_path = os.path.join(self.local_path, self.skymodel)

        with ftp.SLAC(self.slac_path, self.local_path) as slac:
            slac.get('diffuse_info.pkl')
        
        self.df = pd.read_pickle(os.path.join(self.local_path, 'diffuse_info.pkl'))
        
        dir_info = self.shell(f'ls -l {self.local_path}')

        
        #-----------
        self.publishme()

    def normalization_maps(self):
        """
        Galactic Normalization maps

        {fig1}
        """
        gal_norm = self.df.diffuse_normalization.apply(lambda d: d['gal'])
        gal_cube = np.vstack(gal_norm.values)
        hpcube12 =healpix.HPcube(gal_cube, energies=np.logspace(2.125, 3.875, 8))

        self.gal_norm = healpix.HPcube.from_cube(hpcube12, nside=64, sigma=10)

        fig1 = healpix.ait_multiplot(self.gal_norm, vmin=0.9,vmax=1.1, fignum=1);
        #-----------
        self.publishme()

    def residual_maps(self):
        """
        Residual analysis

        Look at the residuals per ROI for the {nume} energy bands.
        First the average

        """
        observed  = np.vstack(self.df.counts.apply(lambda x: x['observed']))
        total    = np.vstack(self.df.counts.apply(lambda x: x['total']))

        energies = self.df.iloc[0]['counts']['energies']
        nume = len(energies)
        labels = list(map(lambda x:f'{x/1e3:.3f}'[:4] , energies));# labels
        respct = 100*(observed-total)/total # residuals in %
        rmeans = respct.mean(axis=0)

        fig1, ax1 = plt.subplots(figsize=(10,5))
        ax1.semilogx(energies, respct.mean(axis=0), 'o-')
        ax1.grid(alpha=0.5);

        t = healpix.HPcube(respct/rmeans, energies )
        residual_cube = healpix.HPcube.from_cube(t, nside=64, sigma=5)
        #--------------
        self.publishme()


# class Summarize(DocPublisher):
#     """

#     title UW All-sky model {version} Summary

#     author Toby Burnett  

#     sections setup

#     slac_path: '/nfs/farm/g/glast/g/catalog/pointlike/skymodels/'
#     local_path: '/tmp/skymodels/'
#     """

#     def __init__(self, **kwargs):
#         super().__init__(self, **kwargs)
#         self.setup()


#     def setup(self):
#         """
#         Setup dowload

#         Will use files found at {self.slac_path}

#         """
#         #----------
#         self.skymodel = self.version or  'uw1208-v3'
#         mkey = self.skymodel[:4]

#         year = dict(uw12='P8_12years', uw89='P305_8years', uw90='P8-10years' )[mkey]

#         self.slac_path = os.path.join(self.slac_path, year,self.skymodel)
#         self.local_path = os.path.join(self.local_path, self.skymodel)
#         #self.publishme()

#     def summary(self):
#         """ Summary

#         File loocation: {self.slac_path}

#         """
     
# class Report(DocPublisher):
#     """
#     title: UW model {skymodel} Report

#     author: Toby Burnett

#     sections : introduction test

#     slac_path: '/nfs/farm/g/glast/g/catalog/pointlike/skymodels/'
#     local_path: '/tmp/skymodels/'

#     decorator_path: https://glast-ground.slac.stanford.edu/Decorator/exp/Fermi/Decorate/groups/catalog/pointlike/skymodels/{}/plots/index.html?skipDecoration
#     """
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.setup()
    
#     def setup(self):
#         self.skymodel = self.version or  'uw1208-v3'
#         mkey = self.skymodel[:4]

#         year = dict(uw12='P8_12years', uw89='P305_8years', uw90='P8-10years' )[mkey]

#         self.slac_path = os.path.join(self.slac_path, year,self.skymodel)
#         self.local_path = os.path.join(self.local_path, self.skymodel)

#         self.decorator = self.decorator_path.format(year+'/'+self.skymodel)

#         with ftp.SLAC(self.slac_path, self.local_path) as slac:
#             print(f'loading folders:  ', end='')
#             for folder in 'sources counts config environment localization'.split():
#                 print(f'.. {folder} ', end='')
#                 slac.get(f'plots/{folder}/*')
#             print()

#     def jpg_name(self, plot_name):
#         return os.path.join(self.local_path, 'plots',  f'{plot_name}_{self.skymodel}.jpg')

#     def introduction(self):
#         """
#         Introduction

#         This a a report on the UW all-sky model {self.skymodel}. 
#         I compile plots from the SLAC folder contain all its files:
        
#          `{self.slac_path}`
        
#         The plots are directly accessible from the 
#         [decorator site](self.decorator).

#         <br>Local path ({self.local_path}) contents:
#         {dir_info}

#         """


#         dir_info = self.shell(f'ls -l {self.local_path}/plots')

#         #---------
#         self.publishme()

#     def test(self):
#         """Test plots
        
#         {chisq}

#         {residual_maps}
#         """
#         chisq = self.image( self.jpg_name('counts/chisq_plots'))
#         residual_maps = self.image(self.jpg_name('counts/residual_maps'))
#         #--------------
#         self.publishme()