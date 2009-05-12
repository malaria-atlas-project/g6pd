# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################


import numpy as np
import pymc as pm
import gc
from map_utils import basic_spatial_submodel, invlogit

__all__ = ['make_model','postproc','f_name','x_name','nugget_name','f_has_nugget','metadata_keys']
    
def make_model(s_obs,a_obs,lon,lat,from_ind,covariate_values):
    """
    """
    
    locs = [(lon[0], lat[0]))]
    fi = [0]
    for i in xrange(1,len(lon)):

        # If repeat location, add observation
        loc = (lon[i], lat[i])
        if loc in locs:
            fi.append(locs.index(loc))

        # Otherwise, new obs
        else:
            locs.append(loc)
            fi.append(max(fi)+1)
    fi = np.array(fi)
    ti = [np.where(fi == i)[0] for i in xrange(max(fi)+1)]

    lon = np.array(locs)[:,0]
    lat = np.array(locs)[:,1]
    

    # V = pm.Exponential('V',.1,value=1.)
    V = pm.OneOverX('V',value=1.)

    init_OK = False
    while not init_OK:
        try:        
            # Space-time component
            sp_sub = basic_spatial_submodel(lon, lat, covariate_values)    

            # The field evaluated at the uniquified data locations            
            f = pm.MvNormalCov('f', sp_sub['M_eval'], sp_sub['C_eval'])
            
            init_OK = True
        except pm.ZeroProbability, msg:
            print 'Trying again: %s'%msg
            init_OK = False
            gc.collect()
        
    # The field plus the nugget
    eps_p_f = pm.Normal('eps_p_f', f[from_ind], 1./V)
    
    s = pm.InvLogit('s',eps_p_f)

    data = pm.Binomial('data', s_obs + a_obs, s, value=s_obs, observed=True)

    out = locals()
    out.pop('sp_sub')
    out.update(sp_sub)

    return out
    
# Stuff mandated by the new map_utils standard

# f_name = 'f'
# x_name = 'logp_mesh'
# f_has_nugget = False
# nugget_name = 'V'

f_name = 'eps_p_f'
x_name = 'data_mesh'
f_has_nugget = True
nugget_name = 'V'
metadata_keys = ['data_mesh','from_ind','to_ind']


postproc = invlogit