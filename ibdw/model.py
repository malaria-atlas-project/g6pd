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
from ibdw import cut_matern, cut_gaussian

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

# constrained = True
# threshold_val = 0.01
# max_p_above = 0.00001

def nested_covariance_fn(x,y, amp, amp_short_frac, scale_short, scale_long, diff_degree, symm=False):
    """
    A nested covariance funcion with a smooth, anisotropic long-scale part
    and a rough, isotropic short-scale part.
    """
    amp_short = amp*np.sqrt(amp_short_frac)
    amp_long = amp*np.sqrt(1-amp_short_frac)
    out = cut_matern(x,y,amp=amp_short,scale=scale_short,symm=symm,diff_degree=diff_degree)
    long_part = cut_gaussian(x,y,amp=amp_long,scale=scale_long,symm=symm)
    out += long_part
    return out

def ncf_diag(x, amp, *args, **kwds):
    return amp**2*np.ones(x.shape[:-1])
    
nested_covariance_fn.diag_call = ncf_diag

def mean_fn(x,m):
    return pm.gp.zero_fn(x)+m

def make_model(lon,lat,input_data,covariate_keys,pos,neg):
    """
    This function is required by the generic MBG code.
    """
    
    # How many nuggeted field points to handle with each step method
    grainsize = 10

    # Unique data locations
    data_mesh, logp_mesh, fi, ui, ti = uniquify(lon,lat)
    
    s_hat = (pos+1.)/(pos+neg+2.)

    normrands = np.random.normal(size=1000)
        
    init_OK = False
    while not init_OK:
        try:
            # The fraction of the partial sill going to 'short' variation.
            amp_short_frac = pm.Uniform('amp_short_frac',0,1)

            # The partial sill.
            amp = pm.Exponential('amp', .1, value=1.)

            # The range parameters. Units are RADIANS. 
            # 1 radian = the radius of the earth, about 6378.1 km
            scale_short = pm.Exponential('scale_short', .1, value=.08)
            scale_long = pm.Exponential('scale_long', .1, value=.9)

            @pm.potential
            def scale_constraint(s=scale_long):
                if s>1:
                    return -np.inf
                else:
                    return 0

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
            V = pm.Exponential('V', .1, value=.2)
            @pm.potential
            def V_constraint(V=V):
                if V<.1:
                    return -np.inf
                else:
                    return 0

            m = pm.Uninformative('m',value=-25)
            @pm.deterministic(trace=False)
            def M(m=m):
                return pm.gp.Mean(mean_fn, m=m)
                
            ceiling = pm.Beta('ceiling',10,50,value=.2)
            
            # if constrained:
            #     @pm.potential
            #     def pripred_check(m=m,amp=amp,V=V,ceiling=ceiling,normrands=np.random.normal(size=1000)):
            #         sum_above = np.sum(pm.flib.invlogit(normrands*np.sqrt(amp**2+V)+m)*ceiling>threshold_val)
            #         if float(sum_above) / len(normrands) <= max_p_above:
            #             return 0.
            #         else:
            #             return -np.inf

            # Create the covariance & its evaluation at the data locations.
            facdict = dict([(k,1.e6) for k in covariate_keys])
            facdict['m'] = 0
            @pm.deterministic(trace=False)
            def C(amp=amp, amp_short_frac=amp_short_frac, scale_short=scale_short, scale_long=scale_long, diff_degree=diff_degree, ck=covariate_keys, id=input_data, ui=ui, facdict=facdict):
                """A covariance function created from the current parameter values."""
                eval_fn = CovarianceWithCovariates(nested_covariance_fn, id, ck, ui, fac=facdict)
                return pm.gp.FullRankCovariance(eval_fn, amp=amp, amp_short_frac=amp_short_frac, scale_short=scale_short, 
                            scale_long=scale_long, diff_degree=diff_degree)

            sp_sub = pm.gp.GPSubmodel('sp_sub', M, C, logp_mesh, tally_f=False)
                
            init_OK = True
        except pm.ZeroProbability:
            init_OK = False
            cls,inst,tb = sys.exc_info()
            print 'Restarting, message %s\n'%inst.message

    # Make f start somewhere a bit sane
    sp_sub.f_eval.value = sp_sub.f_eval.value - np.mean(sp_sub.f_eval.value)

    # Loop over data clusters
    eps_p_f_d = []
    s_d = []
    data_d = []

    for i in xrange(len(pos)/grainsize+1):
        sl = slice(i*grainsize,(i+1)*grainsize,None)        
        if len(pos[sl])>0:
            # Nuggeted field in this cluster
            eps_p_f_d.append(pm.Normal('eps_p_f_%i'%i, sp_sub.f_eval[fi[sl]], 1./V, value=pm.logit(s_hat[sl]), trace=False))            

            # The allele frequency
            s_d.append(pm.Lambda('s_%i'%i,lambda lt=eps_p_f_d[-1], ceiling=ceiling: invlogit(lt)*ceiling,trace=False))

            # The observed allele frequencies
            data_d.append(pm.Binomial('data_%i'%i, pos[sl]+neg[sl], s_d[-1], value=pos[sl], observed=True))
    
    # The field plus the nugget
    @pm.deterministic
    def eps_p_f(eps_p_fd = eps_p_f_d):
        """Concatenated version of eps_p_f, for postprocessing & Gibbs sampling purposes"""
        return np.hstack(eps_p_fd)
            
    return locals()
