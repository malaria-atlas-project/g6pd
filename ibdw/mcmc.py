import numpy as np
from model import *
from pylab import csv2rec
import pymc as pm
from map_utils import FieldStepper
import os

fname = '../data260409.csv'
# landmass = 'Africa'
# 
# landmasses = {'Eurasia': ['Europe','Asia','Oceania'],
#                 'Africa': ['Africa'],
#                 'America': ['America']}

data = csv2rec(fname)
# fdata = data[np.where(np.sum([data.continent == c for c in landmasses[landmass]],axis=0))]
fdata = data
locs = []
from_ind = []
locs = [(float(fdata[0].long), float(fdata[0].lat))]
from_ind = [0]
for i in xrange(1,len(fdata)):
    row = fdata[i]
    
    # If repeat location, add observation
    loc = (float(row.long), float(row.lat))
    if loc in locs:
        from_ind.append(locs.index(loc))
    
    # Otherwise, new obs
    else:
        locs.append(loc)
        from_ind.append(max(from_ind)+1)
from_ind = np.array(from_ind)
to_ind = [np.where(from_ind == i)[0] for i in xrange(max(from_ind)+1)]

lon = np.array(locs)[:,0]
lat = np.array(locs)[:,1]

# TODO TOMORROW: specialize f.logp_fun to print what it's returning before returning.

M=pm.MCMC(make_model(fdata['as'].data,fdata['as'].data + fdata.n.data*2.,lon,lat,from_ind,{}), db='hdf5', dbname=os.path.basename(fname)+'.hdf5', complevel=1)
M.use_step_method(pm.AdaptiveMetropolis, list(M.stochastics -set([M.f, M.eps_p_f])), verbose=0, delay=50000)
for s in M.stochastics | M.deterministics | M.potentials:
    s.verbose = 0
M.use_step_method(FieldStepper, M.f, 1./M.V, M.V, M.C_eval, M.M_eval, M.logp_mesh, M.eps_p_f, to_ind, jump_tau = False)
M.isample(500000,0,100, verbose=0)
# from pylab import *
# plot(M.trace('f')[:])
# M.use_step_method(FieldStepper, M.f, 1./M.V, M.V, M.C_eval, M.M_eval, M.logp_mesh, M.eps_p_f, ti, jump_tau=False)