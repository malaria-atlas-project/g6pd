# from mcmc import *
from model import *
from generic_mbg import invlogit, FieldStepper
from cut_geographic import hemisphere, cut_geographic

# Stuff mandated by the new map_utils standard


diag_safe = False
f_name = 'eps_p_f'
x_name = 'data_mesh'
f_has_nugget = True
nugget_name = 'V'
metadata_keys = ['fi','ti','ui']
postproc = invlogit
step_method_orders = {'f':(FieldStepper, )}
non_cov_columns = {'lo_age': 'int', 'up_age': 'int', 'pos': 'float', 'neg': 'float'}