from __future__ import division
from pylab import *
from pymc import *
from numpy import *
from tables import *
from map_utils import *

# Pull out random points at which to evaluate difference in means
def sample_pts(pts_per_class, n_samps, n_trials, hf):
    wheres = []
    pts = []
    ns = []
    for i in xrange(5):
        wheres.append(where(lys_surf==i+1))
        wts = cos(hf.root.lat[::10][wheres[-1][0]]*np.pi/180.)
        wts /= wts.sum()
        n_here = len(wheres[-1][0])
        ind = list(set(rcategorical(wts, size=pts_per_class)))
        pts.append(vstack((hf.root.long[::10][wheres[-1][1][ind]], hf.root.lat[::10][wheres[-1][0][ind]])))
        ns.append(len(ind))
    pts = hstack(pts).T*pi/180.
    breaks = concatenate(([0],cumsum(ns)))
    return pts, breaks


pts_per_class = 100
n_samps = 100
n_trials = 100

hf = openFile('lyzsalb.hdf5')
lys_surf = hf.root.data[::10,::10]

# TODO: Sample points _geographically_ uniformly, meaning downweight pixels by Jacobian
# TODO: Fracking CSV's for violin plots

pts = []
breaks = []
for i in xrange(n_trials):
    tup = sample_pts(pts_per_class, n_samps, n_trials, hf)
    pts.append(tup[0])
    breaks.append(tup[1])

hf.close()    


tracefile = openFile('run10')
burn = 10000
mciter = len(tracefile.root.chain0.PyMCsamples)

samps = list(set(randint(burn,mciter,size=n_samps)))
n_samps = len(samps)

means = [[] for i in xrange(n_trials)]
pixsamps = [[] for i in xrange(5)]


data_mesh = tracefile.root.metadata.data_mesh[:]
prior_mean_variance = tracefile.root.metadata.covariates[0]['m'][1]
for ii in xrange(len(samps)):
    print ii
    i=samps[ii]
    M = tracefile.root.chain0.group0.M[i]
    C = tracefile.root.chain0.group0.C[i]
    
    V = tracefile.root.chain0.PyMCsamples.cols.V[i]
    f = tracefile.root.chain0.PyMCsamples.cols.eps_p_f[i]
    
    M_input = M(data_mesh)
    
    C_input = C(data_mesh, data_mesh)
    C_input += V*eye(data_mesh.shape[0])
    C_input += ones(C_input.shape)*prior_mean_variance
    try:
        S_input = linalg.cholesky(C_input)
    except linalg.LinAlgError:
        print 'Puking on S_input.'
        continue
    
    for k in xrange(n_trials):
        print '\t',k
        M_pred = M(pts[k])
        C_cross = C(data_mesh, pts[k])
        C_cross += ones(C_cross.shape)*prior_mean_variance
        SC_cross = gp.trisolve(S_input,C_cross,uplo='L',inplace=True)
    
        M_out = M_pred + asarray(dot(SC_cross.T,gp.trisolve(S_input, (f-M_input), uplo='L'))).squeeze()
        C_out = C(pts[k],pts[k])
        C_out += V*eye(C_out.shape[0])
        C_out += ones(C_out.shape)*prior_mean_variance
        C_out -= dot(SC_cross.T, SC_cross)
        try:
            S_out = asarray(linalg.cholesky(C_out))
        except linalg.LinAlgError:
            print 'Puking on S_out'
            continue
        
        samp_out = invlogit(M_out + dot(S_out, normal(size=M_out.shape)))
        
        if random.random()<1./n_trials:
            pixsamp = invlogit(M_out + asarray(sqrt(diag(C_out)))*np.random.normal(size=M_out.shape))
            for l in xrange(len(breaks[k])-1):
                pixsamps[l] = concatenate((pixsamps[l], pixsamp[breaks[k][l]:breaks[k][l+1]]))
            

        means[k].append([mean(samp_out[breaks[k][j]:breaks[k][j+1]]) for j in xrange(len(breaks[k])-1)])
    
# means = np.array(means)
for k in xrange(n_trials):
    means[k] = np.array(means[k])
    
diffprobs = [mean(diff(mm, axis=1)>0,axis=0) for mm in means]
mean_diffprobs = mean(diffprobs, axis=0)
se_diffprobs = std(diffprobs, axis=0)/sqrt(n_trials)

meanprevs = array([mean(mm,axis=0) for mm in means])
pixsamps = array(pixsamps).T

savetxt('meanprevs.csv',meanprevs,delimiter=',')
savetxt('pixsamps.csv',pixsamps,delimiter=',')
savetxt('mean_diffprobs.csv',mean_diffprobs,delimiter=',')
savetxt('se_diffprobs.csv',se_diffprobs,delimiter=',')