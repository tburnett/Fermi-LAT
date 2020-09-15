import os, sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import colors
from astropy.io import fits
import healpy

from . skydir import SkyDir


class HealpixCube(object): 

    def __init__(self, filename):
        self.fullfilename = filename
        try:
            try:
                self.hdulist = hdus = fits.open(self.fullfilename)
            except Exception  as msg:
                raise Exception('FITS: Failed to open {}: {}'.format(self.fullfilename, msg))

            if len(hdus)==2:
                self.energies = []
            else:
                if hdus[2].columns[0].name=='CHANNEL':
                    # binned format: assume next 2 columns are min, max and use geometric mean
                    emin,emax = [hdus[2].data.field(i) for i in (1,2)]
                    self.energies = np.sqrt(emin*emax)
                else:
                    self.energies = hdus[2].data.field(0)
            self.vector_mode = len(hdus[1].columns)==1
            if self.vector_mode:
                # one vector column, expect 2d array with shape (12*nside**2, len(energies))
                self.spectra = hdus[1].data.field(0)
                self.nside = int(np.sqrt(self.spectra.shape[0]/12.))
                assert self.spectra.shape[1]==len(self.energies), 'shape inconsistent with number of energies'
            else:
                # one column per energy: expect len(energies) columns
                hdu1 = hdus[1]
                if len(self.energies)>0:
                    assert len(hdu1.columns)==len(self.energies) , 'wrong number of columns'
                self.data = hdu1.data
                self.nside = int(np.sqrt(self.data.field(0).flatten().shape[0]/12.))
                self.bunit = hdu1.header.get('BUNIT', '')
                assert hdu1.header.get('ORDERING','RING')=='RING', 'Wrong ordering'
                assert hdu1.header.get('COORDSYS', 'GAL')=='GAL', 'Wrong coordsys'
            

        except Exception as msg:
            print(f'bad file or unexpected FITS format, file {filename}: {msg}')
            raise
        hdus.close() 
        #self.logeratio = np.log(self.energies[1]/self.energies[0])
        self.loge= np.log(self.energies)
        if len(self.energies)>0: self.set_energy(1000.)

    def __getitem__(self, index):
        return self.data.field(index)
    def __len__(self):
        return len(self.data.columns)
    def __iter__(self): # iterate over all keys
        for index in range(len(self)):
            yield self[index]

    def set_energy(self, energy): 
        # set up logarithmic interpolation

        self.energy=energy
        #r = np.log(energy/self.energies[0])/self.logeratio
        # get the pair of energies
        if energy< self.energies[0]: i=0
        elif energy>self.energies[-1]: i= len(self.energies)-2
        else:
            i = np.where(self.energies>=energy)[0][0]-1
         
        a,b = self.loge[i], self.loge[i+1]
        self.energy_index = i #= max(0, min(int(r), len(self.energies)-2))
        self.energy_interpolation = (np.log(energy)-a)/(b-a)
        if self.vector_mode:
            self.eplane1 = self.spectra[:,i]
            self.eplane2 = self.spectra[:,i+1]
        else:
            self.eplane1 = np.ravel(self.data.field(i))
            self.eplane2 = np.ravel(self.data.field(i+1))

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
            ret = np.exp( np.log(u) * (1-a) + np.log(v) * a      )
        ### Maybe modify these to accoint for arrays
        # assert np.isfinite(ret), 'Not finite for {} at {} MeV, {},{},{}'.format(skydir, self.energy, a, u,v)
        # if ret<=0:
        #     #print 'Warning: FLux not positive at {} for {:.0f} MeV a={}'.format(skydir, self.energy,a)
        #     ret = 0
        return ret

    def column(self, energy):
        """ return a full HEALPix-ordered column for the given energy
        """
        self.set_energy(energy)
        a = self.energy_interpolation
        if a<0.002:
            return self.eplane1
        elif a>0.998:
            return self.eplane2
        return np.exp( np.log(self.eplane1) * (1-a) 
             + np.log(self.eplane2) * a      )


    def ait_plot(self, energy, **kwargs):
        ait_plot(self, energy, **kwargs)


def ait_plot(mapable, pars=[],
        title='',
        fig=None, 
        pixelsize:'pixel size in deg'=1, 
        projection='aitoff',
        cmap='jet', 
        vmin=None, vmax=None, 
        log=False,
        colorbar=True,
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
        cb = plt.colorbar(im, ax=ax, **cb_kw) 
    ax.grid(color='grey') 
    ax.text( 0.02, 0.95, title, transform=ax.transAxes)

