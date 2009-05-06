from map_utils import *
import tables as tb
import numpy as np

mask_name = '10kbuf5kres.asc'

x, unmasked = asc_to_locs(mask_name,thin=2, bufsize=3)    
hf = tb.openFile('ibd_loc_all_030509.csv.hdf5')
ch = hf.root.chain0
meta = hf.root.metadata

def finalize(prod, n):
    mean = prod[mean_reduce] / n
    var = prod[var_reduce] / n - mean**2
    std = np.sqrt(var)
    std_to_mean = std/mean
    return {'mean': mean, 'var': var, 'std': std, 'std-to-mean':std_to_mean}

products = hdf5_to_samps(ch,meta,x,1000,400,100000,[mean_reduce, var_reduce], 'V', invlogit, finalize)

mean_surf = vec_to_asc(products['mean'],mask_name,'ihd-mean.asc',unmasked)
std_surf = vec_to_asc(products['std'],mask_name,'ihd-std.asc',unmasked)
std_mean_surf = vec_to_asc(products['std-to-mean'],mask_name,'ihd-std-to-mean.asc',unmasked)
