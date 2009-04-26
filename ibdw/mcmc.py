import numpy as np
from model import *
from pylab import csv2rec
import pymc as pm

fname = '../data260409.csv'
landmass = 'Africa'

landmasses = {'Eurasia': ['Europe','Asia','Oceania'],
                'Africa': ['Africa'],
                'America': ['America']}

data = csv2rec(fname)[::10]
fdata = data[np.where(np.sum([data.continent == c for c in landmasses[landmass]],axis=0))]

asd = fdata['as'].data
    
s_obs = asd
a_obs = asd + fdata.n.data*2.

M=pm.MCMC(make_model(s_obs,a_obs,fdata.long,fdata.lat,{}))