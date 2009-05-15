# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################


import numpy as np
import pymc as pm
import gc
from map_utils import *

__all__ = ['make_model','postproc','f_name','x_name','nugget_name','f_has_nugget','metadata_keys','step_method_orders','diag_safe']

def nested_covariance_fn(x,y, amp, amp_short_frac, scale_short, scale_long, inc, ecc, diff_degree):
    """
    A nested covariance funcion with a smooth, anisotropic long-scale part
    and a rough, isotropic short-scale part.
    """
    amp_short = amp*amp_short_frac
    amp_long = amp*(1-amp_short_frac)
    short_part = pm.gp.matern.geo_rad(x,y,diff_degree,amp_short,scale_short)
    long_part = pm.gp.gaussian.aniso_geo_rad(x,y,ecc,inc,amp_long,scale_long)
    return short_part + long_part
    

def ibd_covariance_submodel():
    """
    A small function that creates the mean and covariance object
    of the random field.
    """

    # Anisotropy parameters.
    inc = pm.CircVonMises('inc', 0, 0)
    sqrt_ecc = pm.Uniform('sqrt_ecc', 0, .95)
    ecc = sqrt_ecc**2
    
    # The fraction of the partial sill going to 'short' variation.
    amp_short_frac = pm.Uniform('amp_short_frac',0,1)
    
    # The partial sill.
    amp = pm.Exponential('amp', .1, value=1.)
    
    # The range parameters. Units are RADIANS. 
    # 1 radian = the radius of the earth, about 6378.1 km
    scale_short = pm.Exponential('scale_short', .1, value=.08)
    scale_long = pm.Exponential('scale_long', .1, value=1.)
    
    @pm.potential
    def scale_watcher(short=scale_short,long=scale_long):
        """A constraint: the 'long' scale must be bigger than the 'short' scale."""
        if long>short:
            return 0
        else:
            return -np.Inf
    
    # scale_shift = pm.Exponential('scale_shift', .1, value=.08)
    # scale = pm.Lambda('scale',lambda s=scale_shift: s+.01)
    scale_short_in_km = scale_short*6378.1
    scale_long_in_km = scale_long*6378.1
    
    # This parameter controls the degree of differentiability of the field.
    diff_degree = pm.Uniform('diff_degree', .01, 3)
    
    # The nugget variance.
    V = pm.Exponential('V', .1, value=1.)
    
    # Create the covariance & its evaluation at the data locations.
    @pm.deterministic(trace=True)
    def C(amp=amp, amp_short_frac=amp_short_frac, scale_short=scale_short, scale_long=scale_long, inc=inc, ecc=ecc, diff_degree=diff_degree):
        """A covariance function created from the current parameter values."""
        return pm.gp.FullRankCovariance(nested_covariance_fn, amp=amp, amp_short_frac=amp_short_frac, scale_short=scale_short, 
                    scale_long=scale_long, inc=inc, ecc=ecc, diff_degree=diff_degree)
    
    return locals()
    
    
def make_model(pos,neg,lon,lat,covariate_values,cpus=1):
    """
    This function is required by the generic MBG code.
    """
    
    # How many nuggeted field points to handle with each step method
    grainsize = 10
        
    # Non-unique data locations
    data_mesh = combine_spatial_inputs(lon, lat)
    
    s_hat = (pos+1.)/(pos+neg+2.)
    
    # Uniquify the data locations.
    locs = [(lon[0], lat[0])]
    fi = [0]
    ui = [0]
    for i in xrange(1,len(lon)):

        # If repeat location, add observation
        loc = (lon[i], lat[i])
        if loc in locs:
            fi.append(locs.index(loc))

        # Otherwise, new obs
        else:
            locs.append(loc)
            fi.append(max(fi)+1)
            ui.append(i)
    fi = np.array(fi)
    ti = [np.where(fi == i)[0] for i in xrange(max(fi)+1)]
    ui = np.asarray(ui)

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
            sp_sub = ibd_covariance_submodel()    
            covariate_dict, C_eval = cd_and_C_eval(covariate_values, sp_sub['C'], data_mesh, ui)

            # The field evaluated at the uniquified data locations            
            f = pm.MvNormalCov('f', M_eval, C_eval)
            # Make f start somewhere a bit sane
            f.value = f.value - np.mean(f.value)
        
            # Loop over data clusters
            eps_p_f_d = []
            s_d = []
            data_d = []

            for i in xrange(len(pos)/grainsize+1):
                sl = slice(i*grainsize,(i+1)*grainsize,None)
                # Nuggeted field in this cluster
                eps_p_f_d.append(pm.Normal('eps_p_f_%i'%i, f[fi[sl]], 1./sp_sub['V'], value=pm.logit(s_hat[sl]),trace=False))

                # The allele frequency
                s_d.append(pm.InvLogit('s_%i'%i,eps_p_f_d[-1],trace=False))

                # The observed allele frequencies
                data_d.append(pm.Binomial('data_%i'%i, pos[sl]+neg[sl], s_d[-1], value=pos[sl], observed=True))
            
            # The field plus the nugget
            @pm.deterministic
            def eps_p_f(eps_p_fd = eps_p_f_d):
                """Concatenated version of eps_p_f, for postprocessing & Gibbs sampling purposes"""
                return np.concatenate(eps_p_fd)
            
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

diag_safe = True
f_name = 'eps_p_f'
x_name = 'data_mesh'
f_has_nugget = True
nugget_name = 'V'
metadata_keys = ['fi','ti','ui']
postproc = invlogit
step_method_orders = {'f':(FieldStepper, )}