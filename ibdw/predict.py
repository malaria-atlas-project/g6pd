from map_utils import *
import tables as tb
import numpy as np

import sys
print sys.argv

# map -d ibd-world -k 10kbuf5kres.asc -f ibd_loc_all_030509.csv_matern.hdf5 -b 1000 -r 20 -t 400 -i 5000

mask_name = '10kbuf5kres.asc'
x, unmasked = asc_to_locs(mask_name,thin=20, bufsize=3)    
hf = tb.openFile('ibd_loc_all_030509.csv_matern.hdf5')
ch = hf.root.chain0
meta = hf.root.metadata

n_bins = 1000
bins = np.linspace(0,1,n_bins)
def binfn(arr,n_bins=n_bins):
    return np.array(arr*n_bins,dtype=int)

q = [.05,.25,.5,.75,.95]
    
hsr = histogram_reduce(bins, binfn)
hsf = histogram_finalize(bins, q, hsr)

def finalize(prod, n):
    mean = prod[mean_reduce] / n
    var = prod[var_reduce] / n - mean**2
    std = np.sqrt(var)
    std_to_mean = std/mean
    out = {'mean': mean, 'var': var, 'std': std, 'std-to-mean':std_to_mean}
    out.update(hsf(prod, n))
    return out

# hdf5_to_samps(chain, metadata, x, burn, thin, total, fns, f_label, pred_cv_dict=None, nugget_label=None, postproc=None, finalize=None)
products = hdf5_to_samps(ch,meta,x,1000,400,5000,[mean_reduce, var_reduce, hsr], 'f', None, 'V', invlogit, finalize)
# products = hdf5_to_samps(ch,meta,x,1000,400,5000,[mean_reduce, var_reduce],None, invlogit, finalize)

# mean_surf = vec_to_asc(products['mean'],mask_name,'ihd-mean.asc',unmasked)
# std_surf = vec_to_asc(products['std'],mask_name,'ihd-std.asc',unmasked)
# std_mean_surf = vec_to_asc(products['std-to-mean'],mask_name,'ihd-std-to-mean.asc',unmasked)

for k,v in products.iteritems():
    print k
    q=vec_to_asc(v,mask_name,'ihd-%s.asc'%k,unmasked)