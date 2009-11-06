# from mcmc import *
from generic_mbg import invlogit, FieldStepper
import pymc as pm
from cut_geographic import cut_geographic, hemisphere
import ibdw
import os
root = os.path.split(ibdw.__file__)[0]
pm.gp.cov_funs.cov_utils.mod_search_path.append(root)

import cg
from cg import *

cut_matern = pm.gp.cov_utils.covariance_wrapper('matern', 'pymc.gp.cov_funs.isotropic_cov_funs', {'diff_degree': 'The degree of differentiability of realizations.'}, 'cut_geographic', 'cg')


from model import *

diag_safe = True
f_name = 'eps_p_f'
x_name = 'data_mesh'
f_has_nugget = True
nugget_name = 'V'
metadata_keys = ['fi','ti','ui']
postproc = invlogit
step_method_orders = {'f':(FieldStepper, )}
non_cov_columns = {'lo_age': 'int', 'up_age': 'int', 'pos': 'float', 'neg': 'float'}