# from mcmc import *
from generic_mbg import invlogit, fast_inplace_mul, fast_inplace_square, fast_inplace_scalar_add
import pymc as pm
from cut_geographic import cut_geographic, hemisphere
import g6pd
import numpy as np
import os
root = os.path.split(g6pd.__file__)[0]
pm.gp.cov_funs.cov_utils.mod_search_path.append(root)

import cg
from cg import *

cut_matern = pm.gp.cov_utils.covariance_wrapper('matern', 'pymc.gp.cov_funs.isotropic_cov_funs', {'diff_degree': 'The degree of differentiability of realizations.'}, 'cut_geographic', 'cg', ampsq_is_diag=True)
cut_gaussian = pm.gp.cov_utils.covariance_wrapper('gaussian', 'pymc.gp.cov_funs.isotropic_cov_funs', {}, 'cut_geographic', 'cg', ampsq_is_diag=True)

nugget_labels = {'sp_sub': 'V'}
obs_labels= {'sp_sub': 'eps_p_f'}
non_cov_columns = {'n_male': 'float', 'male_pos': 'float', 'n_fem': 'float', 'fem_pos': 'float'}

def check_data(input):
    for n, pos in zip(['n_male', 'n_fem'], ['male_pos', 'fem_pos']):
        if np.any(input[n])==0:
            raise ValueError, 'Some sample sizes are zero.'
        if np.all(np.isnan(input[pos])) or np.all(np.isnan(input[n])):
            raise ValueError, 'Some NaNs in input'
        if np.any(input[pos]<0) or np.any(input[pos]>input[n]):
            raise ValueError, 'Some observations are negative.'
        
def male_def(sp_sub, ceiling):
    allele = sp_sub.copy('F')
    allele = invlogit(allele)*ceiling
    return allele

def fem_def_conservative(sp_sub, ceiling):
    hom = male_def(sp_sub, ceiling)
    fast_inplace_mul(hom,hom)
    return hom
    
def hw_hetero(sp_sub, ceiling):
    p = male_def(sp_sub, ceiling)
    q = fast_inplace_scalar_add(-p,1)
    fast_inplace_mul(p,q)
    return 2*p
    
def fem_def(sp_sub, ceiling, a, b):
    homo = male_def(sp_sub, ceiling)
    hetero = hw_hetero(sp_sub, ceiling)
    het_def = pm.rbeta(a,b)
    hetero *= het_def
    return hetero+homo

map_postproc = [male_def, fem_def_conservative, fem_def]

def validate_male(data):
    obs = data.male_pos
    n = data.n_male
    def f(sp_sub, ceiling, n=n):
        return pm.rbinomial(n=n,p=pm.invlogit(sp_sub)*ceiling)
    return obs, n, f
    
def validate_female(data):
    obs = data.fem_pos
    n = data.n_fem
    def f(sp_sub, ceiling, a, b, n=n):
        p = pm.invlogit(sp_sub)*ceiling
        h = pm.rbeta(a,b,size=len(sp_sub))
        p_def = g6pd.p_fem_def(p,h)
        return pm.rbinomial(n=n, p=p)
    return obs, n, f

validate_postproc = [validate_male]
# validate_postproc = [validate_female]

regionlist=['Free','Epidemic','Hypoendemic','Mesoendemic','Hyperendemic','Holoendemic']

def area_male(gc):
    if len(gc)>1:
        raise ValueError, "Got geometry collection containing more than one multipolygon: %s"%gc.keys()
    
    def h(**region):
        return np.array(region.values()[0])

    def f(sp_sub, x, ceiling):
        p = pm.invlogit(sp_sub(x))*ceiling
        return p

    g = {gc.keys()[0]: f}
    
    return h,g

def area_fem_def_conservative(gc):
    if len(gc)>1:
        raise ValueError, "Got geometry collection containing more than one multipolygon: %s"%gc.keys()
    
    def h(**region):
        return np.array(region.values()[0])

    def f(sp_sub, x, ceiling):
        p = pm.invlogit(sp_sub(x))*ceiling
        return p**2

    g = {gc.keys()[0]: f}
    
    return h,g

def area_fem_def(gc):
    if len(gc)>1:
        raise ValueError, "Got geometry collection containing more than one multipolygon: %s"%gc.keys()
    
    def h(**region):
        return np.array(region.values()[0])

    def f(sp_sub, x, ceiling, a, b):
        p = pm.invlogit(sp_sub(x))*ceiling
        h = pm.rbeta(a,b,size=len(p))
        return g6pd.p_fem_def(p,h)

    g = {gc.keys()[0]: f}
    
    return h,g

areal_postproc = [area_male, area_fem_def_conservative, area_fem_def]

def mcmc_init(M):
    M.use_step_method(pm.gp.GPParentAdaptiveMetropolis, [M.amp, M.scale, M.diff_degree, M.ceiling, M.a, M.b])
    M.use_step_method(pm.gp.GPEvaluationGibbs, M.sp_sub, M.V, M.eps_p_f)
                    
metadata_keys = ['fi','ti','ui']

from model import *