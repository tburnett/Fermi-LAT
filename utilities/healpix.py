import os, sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import colors
from astropy.io import fits
import healpy

from . skydir import SkyDir


class HPmap(object):
    def __init__(self, map, label='', cblabel=''):
        self.label=label
        self.nside = healpy.get_nside(map)
        self.map = map
        self.cblabel = cblabel

        # set up interpolation

    def __str__(self):
        return f'<{self.__class__.__name__}>, nside {self.nside} label "{self.label}"'
    def __repr__(self): return str(self)

    def __call__(self, skydir:'SkyDir') -> 'value[s]':
        """
        """
        skyindex = skydir.to_healpix(nside=self.nside)
        return self.map[skyindex]

    def ait_plot(self,  **kwargs):
        ait_plot(self, label=self.label, cblabel=self.cblabel, **kwargs)

    def __truediv__(self, other ):
        return self.map / other.map

    def __mul__(self, other):
        return self.map * other.map


class HPcube(HPmap):
    """Implement a spcetral cube with logarithmic interpolation
    """

    def __init__(self, spectral_cube:'2-d array', 
                    energies:'corresponding energies',
                    cblabel='', 
                    energy:'default value'=1000,
                    unit:'units'=''
                     ):
        self.spectra = spectral_cube
        self.nside = healpy.get_nside(self.spectra[0])
        self.maps = list(map(HPmap, self.spectra))
        self.energies = energies
        self.unit = unit
        if not cblabel and unit:
            unit = unit.replace(' ', '\\ ')
            self.cblabel = fr'$\mathrm{{ {unit} }}$'
        else:
            self.cblabel=cblabel
        self.loge = np.log(energies) # for logarithmin interpolation
        self.set_energy(energy)
        if np.std(np.diff(self.energies))<1:
            print('Warning: this is not a spectral cube, which should use logarithmic interpolation')
     
    def __str__(self):
        ee = self.energies
        return f'<{self.__class__.__name__}>: nside {self.nside}, '\
               f' {len(ee)} energies {ee[0]:.2e}-{ee[-1]:.2e} MeV '

    def __getitem__(self, index):
        return self.maps[index]
    def __len__(self):
        return len(self.maps)

    def __iter__(self): # iterate over all keys
        for index in range(len(self)):
            yield self[index]

    def ait_plot(self, index, **kwargs):
        if type(index)==int:
            if index<len(self):
                # small index: just get plane by index
                self[index].ait_plot(  **kwargs)
                return
        # interpret as an energy
        self.set_energy(float(index))
        kw = dict(log=True, cblabel=getattr(self, 'cblabel', ''))
        kw.update(**kwargs)
        ait_plot(self, self.energy, **kw)        

    def set_energy(self, energy): 
        # set up logarithmic interpolation

        self.energy=energy
        # get the pair of energies
        if energy< self.energies[0]: i=0
        elif energy>self.energies[-1]: i= len(self.energies)-2
        else:
            i = np.where(self.energies>=energy)[0][0]-1
         
        a,b = self.loge[i], self.loge[i+1]
        self.energy_index = i #= max(0, min(int(r), len(self.energies)-2))
        self.energy_interpolation = (np.log(energy)-a)/(b-a)

        self.eplane1 = self[i].map
        self.eplane2 = self[i+1].map

    def __call__(self, skydir:'SkyDir', energy=None) -> 'value[s]':
        """
        """
        if energy is not None and energy!=self.energy: 
            self.set_energy(energy)

        skyindex = skydir.to_healpix(nside=self.nside)
        a = self.energy_interpolation
        u, v = self.eplane1[skyindex], self.eplane2[skyindex]
        # avoid interpolation if close to a plane
        if np.abs(a) < 1e-2:  # or v<=0 or np.isnan(v):
            ret = u
        elif np.abs(1-a)< 1e-2: # or u<=0 or np.isnan(u):
            ret = v
        else:
            ret = np.exp( np.log(u) * (1-a) + np.log(v) * a  )
        return ret

    def hpmap(self, energy, label=''):
        """ return an HPmap for the given energy
        """
        self.set_energy(energy)
        a = self.energy_interpolation
        if a<0.002:
            map = self.eplane1
        elif a>0.998:
            map = self.eplane2
        else:
            map = np.exp( np.log(self.eplane1) * (1-a) 
                + np.log(self.eplane2) * a   )
        return HPmap(map, label=label, cblabel=self.cblabel)

    @classmethod
    def from_FITS(cls, filename):
        try:
            hdus = fits.open(filename)
        except Exception  as msg:
            raise Exception('FITS: Failed to open {}: {}'.format(filename, msg))
        try:
            if len(hdus)==2:
                energies = []
                print(f'No energy table in file {filename}')
            else:
                if hdus[2].columns[0].name=='CHANNEL':
                    # binned format: assume next 2 columns are min, max and use geometric mean
                    emin,emax = [hdus[2].data.field(i) for i in (1,2)]
                    energies = np.sqrt(emin*emax)
                else:
                    energies = hdus[2].data.field(0)
            hdu1 = hdus[1]
            data = hdu1.data
            vector_mode = len(hdu1.columns)==1
            
            if vector_mode:
                # one vector column, expect 2d array with shape (12*nside**2, len(energies))
                spectral_cube = hdus[1].data.field(0).T
            else:
                # one column per energy: expect len(energies) columns
                spectral_cube = np.vstack([col.array for col in data.columns])
 
            nside = int(np.sqrt(spectral_cube.shape[0]/12.))
            assert spectral_cube.shape[0]==len(energies), 'shape inconsistent with number of energies'
     
            unit = hdu1.header.get('BUNIT', '')

            assert hdu1.header.get('ORDERING','RING')=='RING', 'Wrong ordering'
            assert hdu1.header.get('COORDSYS', 'GAL')=='GAL', 'Wrong coordsys'
                
        except Exception as msg:
            print(f'bad file or unexpected FITS format, file {filename}: {msg}')
            raise
        hdus.close()
        return cls(spectral_cube, energies, unit=unit) 


    def to_FITS(self, outfile, 
            vector_format:'if True, one-column format'=False, 
            overwrite=True):

        def spectral_table(self):
            
            el = self.energies

            if vector_format:
                columns = [fits.Column(name='spectra', format=f'{len(el)}E',
                            unit=getattr(self, 'unit',''), 
                            array=np.vstack([x.map for x in self]).T)
                            ]
            else:
                columns = [fits.Column(name=f'CHANNEL{i:02d}', format='E', unit='', array=plane.map)
                    for i, plane in enumerate(self)]
                
            table = fits.BinTableHDU.from_columns(columns)
            table.name = 'SKYMAP' 
            # add HEALPix and energy info to the header 
            nside = self.nside

            emin, deltae= el[0], np.log(el[1]/el[0])
            cards = [fits.Card(*pars) for pars in [ 
                    ('FIRSTPIX', 0,             'First pixel (0 based)'),
                    ('LASTPIX',  12*nside**2, 'Last pixel (0 based)'),
                    ('MAPTYPE',  'Fullsky' , ''  ),
                    ('PIXTYPE',  'HEALPIX',      'Pixel algorithm',),
                    ('ORDERING', 'RING',         'Ordering scheme'),
                    ('NSIDE' ,   nside,    'Resolution Parameter'),
                    ('ORDER',    int(np.log2(nside)),   'redundant'),
                    ('INDXSCHM', 'IMPLICIT' ,''),
                    ('OBJECT' ,  'FULLSKY', ''),
                    ('NRBINS',   len(el),      'Number of energy bins'),
                    ('COORDSYS', 'GAL', ''),
                    ('EMIN',     emin,           'Minimum energy'  ),
                    ('DELTAE',   deltae,         'Step in energy (log)'),
                    ('BUNIT',    getattr(self,'unit', ''), ''),
                ]]
            for card in cards: table.header.append(card)
            return table

        def energy_table(self):
            column= fits.Column( name='energy', format='E', unit='MeV', array=self.energies)
            table = fits.BinTableHDU.from_columns([column])
            table.name='ENERGIES'
            return table

        def primary(self):
            cards = [fits.Card('COMMENT', 'Written by utilities/healpix.py '),
                    ]
            return fits.PrimaryHDU(header=fits.header.Header(cards))

        def hdu_list(self):
            return [ primary(self),  #primary
                    spectral_table(self),           # this table
                    energy_table(self),
                ]

        hdus = hdu_list(self)

        fits.HDUList(hdus).writeto(outfile, overwrite=overwrite)
        print( f'\nwrote {"vector format" if vector_format else ""} '\
            f'FITS Skymap file, nside={self.nside}, {len(self.energies)} energies, to {outfile}')


    # def ratio(self, other, skydir, energy, label=''):
    #     """return the ratio of this with another cube at the energy
    #     """
    #     return self(skydir,energy) / other(skydir,energy)

class HPcubeOp(HPcube):

    """ Base class for operations 
    """
    def __init__(self, hpmap1, hpmap2):
        self.map1=hpmap1
        self.map2=hpmap2
        self.set_energy(1000.)

    def set_energy(self, energy):
        self.energy=float(energy)


class HPratio(HPcubeOp):
    """Ratio of two HpMap obects
    """    
    def __call__(self, skydir:'SkyDir', 
                energy) -> 'value[s]':
        return self.map1(skydir,float(energy)) / self.map2(skydir,float(energy))


class HPproduct(HPcubeOp):
    """ Product of two HPmap objecs
    """

    def __call__(self, skydir:'SkyDir', 
                energy) -> 'value[s]':
        self.set_energy(energy)
        return self.map1(skydir,self.energy) * self.map2(skydir,self.energy)

def ait_plot(mapable, 
        pars=[],
        label='',        
        title='',
        fig=None, ax=None,
        pixelsize:'pixel size in deg'=1, 
        projection='aitoff',
        cmap='jet', 
        vmin=None, vmax=None, 
        log=False,
        colorbar=True,
        cblabel='',
        cb_kw={}, 
        axes_pos=111,
        axes_kw={},
        ):
    """
    """
    #  
    # healpy.mollview(self.column(energy), **kwargs)

    # code inspired by https://stackoverflow.com/questions/46063033/matplotlib-extent-with-mollweide-projection

    # make a mesh grid
    nx, ny = 360//pixelsize, 180//pixelsize
    lon = np.linspace(-180, 180, nx)
    lat = np.linspace(-90., 90, ny)
    Lon,Lat = np.meshgrid(lon,lat)

    #  an arrary of values corresponding to the grid
    dirs = SkyDir.from_galactic(Lon, Lat)
    arr = mapable(dirs, *np.atleast_1d(pars))

    if ax:
        fig = ax.figure
        assert ax.__class____name__=='AitoffAxesSubplot'
    else:
        fig = plt.figure(figsize=(12,5)) if fig is None else fig
        # this needs to be more flexible
        ax = fig.add_subplot(axes_pos, projection=projection, **axes_kw)

    # reverse longitude sign here for display

    im = ax.pcolormesh(-np.radians(Lon), np.radians(Lat), arr, 
        norm=colors.LogNorm() if log else None,
        cmap=cmap,  vmin=vmin, vmax=vmax)
    ax.set(xticklabels=[], yticklabels=[])
    if colorbar:
        cb_kw.update(label=cblabel)
        cb = plt.colorbar(im, ax=ax, **cb_kw) 
    ax.grid(color='grey')  
    if label:   
        ax.text( 0.02, 0.95, label, transform=ax.transAxes)
    if title:
        plt.suptitle(title, fontsize=12)


def ait_multiplot(mapable, 
        energies, 
        labels, 
        layout:'like 22 for 2x2', 
        fig=None, 
        title=None,         
        **kwargs ):

    fig = fig or plt.figure( figsize=(15,8), num=1)
    kw = dict(fig=fig, **kwargs)
    
    axes_pos = 10*layout+1
    for i, (energy, label) in enumerate(zip(energies, labels)):
        ait_plot(mapable,  energy,  label=label, axes_pos=axex_pos+i, **kw)

    fig.tight_layout()
    if title: fig.suptitle(title, fontsize=16);


class Polyfit(object):
    """ Manage a log polynomral fit to each pixel
    """
    def __init__(self, 
        cubefile, sigsfile, start=0, stop=8, deg=2,limits=(0.5,25)):
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
           