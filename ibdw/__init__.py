# from mcmc import *
from generic_mbg import invlogit, fast_inplace_mul, fast_inplace_square, fast_inplace_scalar_add
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
        
def allele(sp_sub):
    allele = sp_sub.copy('F')
    allele = invlogit(hbs)
    return allele

def hw_homo(sp_sub):
    allele = allele(sp_sub)
    return fast_inplace_square(allele)
    
def hw_hetero(sp_sub):
    p = allele(sp_sub)
    q = fast_inplace_scalar_add(1,-p)
    return 2*fast_inplace_mul(p,q)
    
def hw_any(sp_sub):
    homo = hw_homo(sp_sub)
    hetero = hw_hetero(sp_sub)
    return hetero+homo

map_postproc = [allele, hw_hetero, hw_homo, hw_any]

regionlist=['Free','Epidemic','Hypoendemic','Mesoendemic','Hyperendemic','Holoendemic']

def joint_areal_means(gc, regionlist=regionlist): 

    def h(**kwds):
        return np.array([kwds[r] for r in regionlist])

    g = dict([(k, lambda sp_sub, x, a=v['geom'].area: invlogit(sp_sub(x))) for k,v in gc.iteritems()])
    return h, g
    
def independent_areal_means(gc):
    if len(gc)>1:
        raise ValueError, "Got geometry collection containing more than one multipolygon: %s"%gc.keys()
    
    def h(**region):
        return np.array(region.values()[0])

    g = {gc.keys()[0]: lambda sp_sub, x: invlogit(sp_sub(x))}
    
    return h,g

areal_postproc = [independent_areal_means]

def mcmc_init(M):
    M.use_step_method(pm.gp.GPParentAdaptiveMetropolis, [M.amp, M.amp_short_frac, M.scale_short, M.scale_long, M.diff_degree])
    M.use_step_method(pm.gp.GPEvaluationGibbs, M.sp_sub, M.V, M.eps_p_f)
                    
metadata_keys = ['fi','ti','ui']

from model import *