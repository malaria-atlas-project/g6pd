# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################


import numpy as np
import pymc as pm
import gc
from map_utils import *

__all__ = ['make_model','postproc','f_name','x_name','nugget_name','f_has_nugget','metadata_keys','step_method_orders']

def ibd_spatial_submodel():
    """
    A small function that creates the mean and covariance object
    of the random field.
    """

    # Anisotropy parameters.
    inc = pm.CircVonMises('inc', 0, 0)
    sqrt_ecc = pm.Uniform('sqrt_ecc', 0, .95)
    ecc = pm.Lambda('ecc', lambda s=sqrt_ecc: s**2)
    
    # The partial sill.
    amp = pm.Exponential('amp', .1, value=1.)
    
    # The range parameter. Units are RADIANS. 
    # 1 radian = the radius of the earth, about 6378.1 km
    scale_shift = pm.Exponential('scale_shift', 1./.08, value=.08)
    scale = pm.Lambda('scale',lambda s=scale_shift: s+.01)
    
    # This parameter controls the degree of differentiability of the field.
    diff_degree = pm.Uniform('diff_degree', .01, 3)
    
    # The nugget variance.
    V = pm.Exponential('V', .1, value=1.)
    
    # Create the covariance & its evaluation at the data locations.
    @pm.deterministic(trace=True)
    def C(amp=amp, scale=scale, inc=inc, ecc=ecc, diff_degree=diff_degree):
        return pm.gp.FullRankCovariance(pm.gp.cov_funs.matern.aniso_geo_rad, amp=amp, scale=scale, inc=inc, ecc=ecc, diff_degree=diff_degree)
    
    return locals()
    
def make_model(pos,neg,lon,lat,covariate_values,cpus=1):
    """
    This function is required by the generic MBG code.
    """
        
    # Non-unique data locations
    data_mesh = combine_spatial_inputs(lon, lat)
    
    s_hat = (pos+1.)/(pos+neg+2.)
    
    # Uniquify the data locations.
    locs = [(lon[0], lat[0])]
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

    # Unique data locations
    logp_mesh = combine_spatial_inputs(lon,lat)
    
    # Create the mean & its evaluation at the data locations.
    M, M_eval = trivial_means(logp_mesh)

    init_OK = False
    while not init_OK:
        try:        
            # Space-time component
            sp_sub = ibd_spatial_submodel()    
            covariate_dict, C_eval = cd_and_C_eval(covariate_values, sp_sub['C'], logp_mesh)

            # The field evaluated at the uniquified data locations            
            f = pm.MvNormalCov('f', M_eval, C_eval)
        
            # The field plus the nugget
            eps_p_f = pm.Normal('eps_p_f', f[fi], 1./sp_sub['V'], value=pm.logit(s_hat))

            # The allele frequency
            s = pm.InvLogit('s',eps_p_f)
            
            # The observed allele frequencies
            data = pm.Binomial('data', pos+neg, s, value=pos, observed=True)
            
            init_OK = True
        except pm.ZeroProbability, msg:
            print 'Trying again: %s'%msg
            init_OK = False
            gc.collect()
        

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
metadata_keys = ['data_mesh','fi','ti']
postproc = invlogit
step_method_orders = {'f':(FieldStepper, )}