# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################


import numpy as np
import pymc as pm
import gc
from map_utils import *
from generic_mbg import *
import generic_mbg


__all__ = ['make_model','nested_covariance_fn']

# The parameterization of the cut between western and eastern hemispheres.
#
# t = np.linspace(0,1,501)
# 
# def latfun(t):
#     if t<.5:
#         return (t*4-1)*np.pi
#     else:
#         return ((1-t)*4-1)*np.pi
#         
# def lonfun(t):
#     if t<.25:
#         return -28*np.pi/180.
#     elif t < .5:
#         return -28*np.pi/180. + (t-.25)*3.5
#     else:
#         return -169*np.pi/180.
#     
# lat = np.array([latfun(tau)*180./np.pi for tau in t])    
# lon = np.array([lonfun(tau)*180./np.pi for tau in t])

def nested_covariance_fn(x,y, amp, amp_short_frac, scale_short, scale_long, diff_degree, symm=False):
    """
    A nested covariance funcion with a smooth, anisotropic long-scale part
    and a rough, isotropic short-scale part.
    """
    amp_short = amp*np.sqrt(amp_short_frac)
    amp_long = amp*np.sqrt(1-amp_short_frac)
    out = pm.gp.matern.cut_geographic(x,y,amp=amp_short,scale=scale_short,symm=symm,diff_degree=diff_degree)
    long_part = pm.gp.gaussian.cut_geographic(x,y,amp=amp_long,scale=scale_long,symm=symm,inc=inc,ecc=ecc)
    out += long_part
    return out

def mean_fn(x,m):
    return np.zeros(x.shape[:-1])+m

def ibd_covariance_submodel():
    """
    A small function that creates the mean and covariance object
    of the random field.
    """
    
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
    def C(amp=amp, amp_short_frac=amp_short_frac, scale_short=scale_short, scale_long=scale_long, diff_degree=diff_degree):
        """A covariance function created from the current parameter values."""
        return pm.gp.FullRankCovariance(nested_covariance_fn, amp=amp, amp_short_frac=amp_short_frac, scale_short=scale_short, 
                    scale_long=scale_long, diff_degree=diff_degree)
    
    return locals()
    
    
def make_model(lon,lat,covariate_values,pos,neg,cpus=1):
    """
    This function is required by the generic MBG code.
    """
    
    if np.any(pos+neg==0):
        where_zero = np.where(pos+neg==0)[0]
        raise ValueError, 'Pos+neg = 0 in the rows (starting from zero):\n %s'%where_zero
    
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
    
    # m = pm.Uniform('m',-10,-5)
    m = pm.Uninformative('m',value=-7)
        
    normrands = np.random.normal(size=1000)
        
    # Create the mean & its evaluation at the data locations.
    @pm.deterministic
    def M(m=m):
        return pm.gp.Mean(mean_fn, m=m)

    @pm.deterministic
    def M_eval(M=M):
        return M(logp_mesh)



    init_OK = False
    while not init_OK:
        try:        
            # Space-time component
            sp_sub = ibd_covariance_submodel()    
            
            @pm.potential
            def pripred_check(m=m,amp=sp_sub['amp'],V=sp_sub['V'],normrands=normrands):
                sum_above = np.sum(pm.flib.invlogit(normrands*np.sqrt(amp+V)+m)>.017)
                if float(sum_above) / len(normrands) <= 1.-.79:
                    return 0.
                else:
                    return -np.inf
            
            
            covariate_dict, C_eval = cd_and_C_eval(covariate_values, sp_sub['C'], data_mesh, ui, fac=0)
            
            @pm.deterministic
            def S_eval(C_eval=C_eval):
                try:
                    return np.linalg.cholesky(C_eval)
                except np.linalg.LinAlgError:
                    return None
                    
            @pm.potential
            def fr_check(S_eval=S_eval):
                return -np.inf if S_eval is None else 0

            # The field evaluated at the uniquified data locations            
            f = pm.MvNormalChol('f', M_eval, S_eval)
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
                s_d.append(pm.Lambda('s_%i'%i,lambda lt=eps_p_f_d[-1]: invlogit(lt),trace=False))

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
