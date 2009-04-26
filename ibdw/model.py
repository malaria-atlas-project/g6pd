# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################


import numpy as np
import pymc as pm
import gc
from map_utils import basic_spatial_submodel

__all__ = ['make_model']
    
def make_model(s_obs,a_obs,lon,lat,covariate_values,cpus=1):
    """
    d : transformed ('gaussian-ish') data
    lon : longitude
    lat : latitude
    t : time
    covariate_values : {'ndvi': array, 'rainfall': array} etc.
    cpus : int
    """
        
        # Space-time component
        st_sub = basic_spatial_submodel(lon, lat, covariate_values, cpus)
            
        # The field plus the nugget
        eps_p_f = pm.Normal('eps_p_f', st_sub['f'], st_sub['V'])
        
        s = pm.InvLogit('s',eps_p_f)
    
        data = pm.Binomial('data', s_obs + a_obs, s, value=s_obs)

    return locals()