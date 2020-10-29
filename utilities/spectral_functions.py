"""Spectral energy distributions
"""

import numpy as np
import matplotlib.pyplot as plt

class LogParabola(object):
    
    def __init__(self, 
            pars:'parameters: n0,alpha,beta,e_break  '):
        self.p = pars
    
    def __call__(self, e:'energy in MeV'
        )->'energy flux in eV / s / cm^2':
        """
        """
        n0,alpha,beta,e_break = self.p
        x = np.log(e_break/e)
        y = (alpha - beta*x)*x
        return e**2 * 1e6 * n0 * np.exp(y)
    
    def alpha(self, e):
        n0,alpha,beta,e_break = self.p
        return alpha + beta * np.log(e/e_break)
        
    @property
    def peak(self)->"Energy at peak":

        n0,alpha,beta,e_break = self.p
        return e_break * np.exp((2-alpha)/beta)
    
    def sed(self, name='', ax=None, 
                xlim=(100, 4e4),  
                ylim=(0.06,20)):
        if ax is None:
            fig,ax = plt.subplots(figsize=(3.5,3.5))
        else: fig = ax.figure
        
        x = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]) ) 
        ax.loglog(x, self(x), '-r', lw=2)
        ax.set(ylim=ylim)
        ax.grid(alpha=0.5)
        if name:
            ax.text(0.1, 0.9, name, transform=ax.transAxes, fontsize=12)
        return fig

    
