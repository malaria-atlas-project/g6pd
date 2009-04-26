import numpy as np
from model import *
from pylab import csv2rec
import pymc as pm

fname = '../data260409.csv'
landmass = 'Africa'

landmasses = {'Eurasia': ['Europe','Asia','Oceania'],
                'Africa': ['Africa'],
                'America': ['America']}

data = csv2rec(fname)
fdata = data[np.where(np.sum([data.continent == c for c in landmasses[landmass]],axis=0))]
locs = []
s_obs = {}
a_obs = {}
for row in fdata:
    
    # If repeat location, add observation
    loc = (float(row.long.data), float(row.lat.data))
    if loc in locs:
        s_obs[loc] += row['as'].data
        a_obs[loc] += row['as'].data + row.n.data*2.
    
    # Otherwise, new obs
    else:
        locs.append(loc)
        s_obs[loc] = row['as'].data
        a_obs[loc] = row['as'].data + row.n.data*2.

lon = np.array(locs)[:,0]
lat = np.array(locs)[:,1]
s_obs = np.array([s_obs[loc] for loc in locs])
a_obs = np.array([a_obs[loc] for loc in locs])


M=pm.MCMC(make_model(s_obs,a_obs,lon,lat,{}))
M.use_step_method(pm.AdaptiveMetropolis, list(M.stochastics))