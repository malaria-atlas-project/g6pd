import pymc as pm
import imp
imp.load_module('cut_geographic', *imp.find_module('cut_geographic',pm.gp.cov_funs.cov_utils.mod_search_path))
from cut_geographic import cut_geographic as cg

def cut_geographic(D,x,y,nx,ny,cmin=0,cmax=-1,symm=0):
    """
    A distance function that separates the Americas from the rest of the world.
    """
    return cg(D,x,y,nx,ny,cmin,cmax,symm)