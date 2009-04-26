# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################


import numpy as np
import pymc as pm
import gc

__all__ = ['make_model','combine_input_data']

def combine_input_data(lon,lat):
    # Convert latitude and longitude from degrees to radians.
    lon = lon*np.pi/180.
    lat = lat*np.pi/180.
    
    # Make lon, lat tuples.
    data_mesh = np.vstack((lon, lat)).T 
    return data_mesh
    
def make_model(s_obs,a_obs,lon,lat,covariate_values,cpus=1):
    """
    d : transformed ('gaussian-ish') data
    lon : longitude
    lat : latitude
    t : time
    covariate_values : {'ndvi': array, 'rainfall': array} etc.
    cpus : int
    """
        
    logp_mesh = combine_input_data(lon,lat)

    # =====================
    # = Create PyMC model =
    # =====================    
    init_OK = False
    while not init_OK:
                
        # Make coefficients for the covariates.
        m_const = pm.Uninformative('m_const', value=0.)
        
        covariate_dict = {}
        for cname, cval in covariate_values.iteritems():
            this_coef = pm.Uninformative(cname + '_coef', value=0.)
            covariate_dict[cname] = (this_coef, cval)

        V = pm.Exponential('V',.1,value=1.)

        inc = pm.CircVonMises('inc', 0,0)

        @pm.stochastic(__class__ = pm.CircularStochastic, lo=0, hi=1)
        def sqrt_ecc(value=.1):
            return 0.
        ecc = pm.Lambda('ecc', lambda s=sqrt_ecc: s**2)

        amp = pm.Exponential('amp',.1,value=1.)

        scale = pm.Exponential('scale',.1,value=1.)
        
        dd = pm.Uniform('dd',.5,3)
            
        # The mean of the field
        @pm.deterministic(trace=True)
        def M(mc=m_const):
            return pm.gp.Mean(lambda x: mc*np.ones(x.shape[0]))
        
        # The mean, evaluated  at the observation points, plus the covariates    
        @pm.deterministic(trace=False)
        def M_eval(M=M, lpm=logp_mesh, cv=covariate_dict):
            out = M(lpm)
            for c in cv.itervalues():
                out += c[0]*c[1]
            return out

        # Create covariance and MV-normal F if model is spatial.   
        try:
            # A Deterministic valued as a Covariance object. Uses covariance my_st, defined above. 
            @pm.deterministic(trace=True)
            def C(amp=amp,scale=scale,inc=inc,ecc=ecc):
                return pm.gp.FullRankCovariance(pm.gp.cov_funs.matern.aniso_geo_rad, amp=amp, scale=scale, inc=inc, ecc=ecc, diff_degree=dd,n_threads=cpus)

            # The evaluation of the Covariance object, plus the nugget.
            @pm.deterministic(trace=False)
            def C_eval(C=C):
                return C(logp_mesh, logp_mesh)
                                            
            # The field evaluated at the uniquified data locations
            f = pm.MvNormalCov('f',M_eval,C_eval)            
            
            init_OK = True
        except pm.ZeroProbability, msg:
            print 'Trying again: %s'%msg
            init_OK = False
            gc.collect()
            
        # The field plus the nugget
        eps_p_f = pm.Normal('eps_p_f', f, V)
        
        s = pm.InvLogit('s',eps_p_f)
    
        data = pm.Binomial('data', s_obs + a_obs, s, value=s_obs)

    return locals()