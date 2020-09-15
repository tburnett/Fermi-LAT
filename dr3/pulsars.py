"""
"""
import os,sys


import os, sys
import shutil
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import pysftp as sftp

from jupydoc import DocPublisher

__docs__ = ['Weak']


class Weak(DocPublisher):
    """
    title: Properties of undetected pulsars
    author: Toby Burnett
    sections: load_files

    """
    server =     'rhel6-64.slac.stanford.edu'
    username =   'burnett'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        models = dict(uw90='P8_10years/',  uw12 = 'P8_12years/')
        v = self.version
        self.skymodel= models[v[:4]]+v
        self.model_path = '/nfs/farm/g/glast/g/catalog/pointlike/skymodels/'+self.skymodel

    def get_slac_file(self, f:'SLAC filename',  
                        t:'local filename'):
        """
        """
        fn = os.path.join(self.tmp, t)
        d,_ = os.path.split(fn)
        os.makedirs(d, exist_ok=True)
        if os.path.isfile(fn): return fn
        try:
            srv = sftp.Connection(self.server, self.username)
        except:
            print(f'Failed to connect to {self.username}@{self.server}', file=sys.stderr)
            raise
        srv.chdir(self.model_path)
        try:
            srv.get( f, fn)
        except Exception as msg:
            print(f'Failed to copy file {f} to {fn}: {msg}')
        srv.close()

    def load_files(self, reload=False):
        """Set up files

        #### Files copied from SLAC to {tmp}, {isloaded} loaded.
      
        {tmpfiles}

        Loaded DataFrame with {npsr} pulsars.
        """
        self.tmp = tmp = '/tmp/dr3/'+self.skymodel

        if os.path.isdir(tmp) and not reload:
            isloaded = 'already'
        else:
            try:
                srv = sftp.Connection(self.server, self.username)
            except:
                print(f'Failed to connect to {self.username}@{self.server}', file=sys.stderr)
                raise
            srv.chdir(self.model_path)
            listdir = srv.listdir()
            os.makedirs(tmp, exist_ok=True)  
            srv.get(f'pulsars.csv', tmp+'/pulsars.csv')
            
            srv.close()    
            isloaded = 'just now'

        tmpfiles = self.shell(f'ls -l {tmp}')
        self.df = pd.read_csv(self.tmp+'/pulsars.csv', index_col=0)
        npsr = len(self.df)
        self.publishme()
