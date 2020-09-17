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

    def __init__(self, maps:'list of HPmap objects', 
                    energies:'corresponding energies',
                     cblabel='', 
                     energy:'default value'=1000,
                     **kwargs ):
        self.maps = maps
        self.energies = energies
        self.cblabel=cblabel
        self.loge = np.log(energies) # for logarithmin interpolation
        self.nside = maps[0].nside
        self.__dict__.update(**kwargs)
        self.set_energy(energy)
        if np.std(np.diff(self.energies))<1:
            print('Warning: this is not a spectral cube, which should use logarithmic interpolation')
     
    def __call__(self, skydir:'SkyDir', 
                energy:'energy',
                ) -> 'value[s]':
        """
        """
        skyindex = skydir.to_healpix(nside=self.nside)
        return self.map[skyindex]

    def __str__(self):
        return('\n'.join([str(map) for map in self]))

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
        ait_plot(self, self.energy, **kwargs)        

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
                spectra = hdus[1].data.field(0)
                nside = int(np.sqrt(spectra.shape[0]/12.))
                assert spectra.shape[1]==len(energies), 'shape inconsistent with number of energies'

                maps = []
                for i, col in enumerate(spectra.T):
                    E = energies[i]
                    maps.append(HPmap(col, f'{E:.2f} MeV'))
            else:
                # one column per energy: expect len(energies) columns
                if len(energies)>0:
                    assert len(hdu1.columns)==len(energies) , 'wrong number of columns'
                spectra = np.vstack([col.array for col in data.columns])
       
                nside = int(np.sqrt(data.field(0).flatten().shape[0]/12.))

                    
            # construct colorbar label for Latex
            bunit = hdu1.header.get('BUNIT', '')
            unit = bunit.replace(' ', '\\ ')
            cblabel = fr'$\mathrm{{ {unit} }}$'
            assert hdu1.header.get('ORDERING','RING')=='RING', 'Wrong ordering'
            assert hdu1.header.get('COORDSYS', 'GAL')=='GAL', 'Wrong coordsys'
                
        except Exception as msg:
            print(f'bad file or unexpected FITS format, file {filename}: {msg}')
            raise
        hdus.close()
        return cls(maps, energies, cblabel=cblabel, bunit=bunit) # layer_names, column_names, label, cblabel)


    def to_FITS(self, outfile, overwrite=True):

        def spectral_table(self):
            array = np.vstack([x.map for x in self]).T
            el = self.energies
            column = fits.Column(name='spetra', format=f'{len(el)}E',
                        unit=self.bunit, array=array)
            table = fits.BinTableHDU.from_columns([column])
            table.name = 'SKYMAP' 
            # add HEALPix and energy info to the header 
            nside = self.nside

            emin, deltae= el[0], np.log(el[1]/el[0])
            cards = [fits.Card(*pars) for pars in [ 
                    ('PIXTYPE',  'HEALPIX',      'Pixel algorithm',),
                    ('ORDERING', 'RING',         'Ordering scheme'),
                    ('NSIDE' ,    self.nside,    'Resolution Parameter'),
                    ('FIRSTPIX',  0,             'First pixel (0 based)'),
                    ('LASTPIX',  array.shape[0], 'Last pixel (0 based)'),
                    ('INDXSCHM', 'IMPLICIT' ,''),
                    ('OBJECT' ,  'FullSky', ''),                                                            
                    ('COORDSYS', 'GAL', ''),
                    ('NRBINS',   array.shape[1], 'Number of energy bins'),
                    ('EMIN',     emin,           'Minimum energy'  ),
                    ('DELTAE',   deltae,         'Step in energy (log)'),
                ]]
            for card in cards: table.header.append(card)
            return table

        def energy_table(self):
            column= fits.Column( name='MeV', format='E', unit='MeV', array=self.energies)
            table = fits.BinTableHDU.from_columns([column])
            table.name='ENERGIES'
            return table

        def hdu_list(self):
            return [ fits.PrimaryHDU(header=None),  #primary
                    spectral_table(self),           # this table
                    energy_table(self),
                ]

        hdus = hdu_list(self)

        fits.HDUList(hdus).writeto(outfile, overwrite=overwrite)
        print( f'\nwrote FITS Skymap file, nside={self.nside}, {len(self.energies)} energies, to {outfile}')


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
        fig=None, 
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
        plt.suptitle(title)


