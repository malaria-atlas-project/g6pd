# from mcmc import *
from generic_mbg import invlogit
import pymc as pm
from cut_geographic import cut_geographic, hemisphere
import ibdw
import numpy as np
import os
root = os.path.split(ibdw.__file__)[0]
pm.gp.cov_funs.cov_utils.mod_search_path.append(root)

import cg
from cg import *

cut_matern = pm.gp.cov_utils.covariance_wrapper('matern', 'pymc.gp.cov_funs.isotropic_cov_funs', {'diff_degree': 'The degree of differentiability of realizations.'}, 'cut_geographic', 'cg')
cut_gaussian = pm.gp.cov_utils.covariance_wrapper('gaussian', 'pymc.gp.cov_funs.isotropic_cov_funs', {}, 'cut_geographic', 'cg')

nugget_labels = {'sp_sub': 'V'}
obs_labels= {'sp_sub': 'eps_p_f'}
non_cov_columns = {'pos': 'float', 'neg': 'float'}

def check_data(input):
    if np.any(input.pos+input.neg)==0:
        raise ValueError, 'Some sample sizes are zero.'
    if np.any(np.isnan(input.pos)) or np.any(np.isnan(input.neg)):
        raise ValueError, 'Some NaNs in input'
    if np.any(input.pos<0) or np.any(input.neg<0):
        raise ValueError, 'Some negative values in pos and neg'
        
def hbs(sp_sub):
    hbs = sp_sub.copy('F')
    hbs = invlogit(hbs)
    return hbs

map_postproc = [hbs]

def areal_diff(gc): 
    "Difference in areal mean between some areas" 

    def h(Free, Epidemic, Hypoendemic, Mesoendemic, Hyperendemic, Holoendemic):
        "The V is in there just to test"
        return np.diff([Free, Epidemic, Hypoendemic, Mesoendemic, Hyperendemic, Holoendemic])

    g = dict([(k, lambda sp_sub, x, a=v['geom'].area: invlogit(sp_sub(x))) for k,v in gc.iteritems()])
    return h, g

areal_postproc = [areal_diff]

def mcmc_init(M):
    M.use_step_method(pm.gp.GPParentAdaptiveMetropolis, [M.amp, M.amp_short_frac, M.scale_short, M.scale_long, M.diff_degree])
                    
metadata_keys = ['fi','ti','ui']

from model import *