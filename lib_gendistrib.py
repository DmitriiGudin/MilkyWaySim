from __future__ import division
import numpy as np
import math
import inspect
import random
import pymp

# Calculates the bottom and the upper limits of the Monte-Carlo box to use for the distribution generation.
# Output: a tuple of the bottom and the upper limit.
# -----
# func: multi-dimensional density distribution function f(x1,x2...xn).
# left/right_lim: tuples or lists of coordinates of bottom/top function argument limits.
# N: number of samples. Positive integer.
# xFactor: expands the box limits, to make sure the whole distribution is within the box. At least 1.
def est_MC_bounds (func, left_lim, right_lim, N=1000, xFactor=2):
    dim = len(inspect.getargspec(func)[0])
    bot_lim, up_lim = 0, 0
    random.seed()
    for i in np.arange (0, N):
        Roll = np.zeros(dim)
        for j in np.arange (0, dim):
            Roll[j]=random.uniform(left_lim[j],right_lim[j])
        bot_lim = min(bot_lim, func(*Roll))
        up_lim = max(up_lim, func(*Roll))
    return max((up_lim+bot_lim)/2-(up_lim-bot_lim)/2*xFactor,0), max((up_lim+bot_lim)/2+(up_lim-bot_lim)/2*xFactor,0)

# Generates a distribution of N points from a given density distribution.
# Output: a list of N points (each is a tuple of coordinates).
# -----
# func: multi-dimensional density distribution function f(x1,x2...xn).
# left/right_lim: tuples or lists of coordinates of bottom/top function argument limits.
# N: number of points to generate. Positive integer.
# lim_N_samples: number of samples for the MC box limits determination. Positive integer.
# lim_xFactor: MC box limits expansion factor. At least 1.
def produce_distr (func, left_lim, right_lim, N, N_CORES=1, lim_N_samples=1000, lim_xFactor=2):
    dim = len(inspect.getargspec(func)[0])
    bot_lim, up_lim = est_MC_bounds (func, left_lim, right_lim, lim_N_samples, lim_xFactor)
    random.seed()
    output_list=pymp.shared.list()
    with pymp.Parallel(N_CORES) as p:
        for i in p.range (0, N):
            while(True):
                Roll, Roll_value = np.zeros(dim), 0  
                for j in np.arange (0, dim):
                    Roll[j] = random.uniform(left_lim[j],right_lim[j])
                Roll_value = random.uniform(bot_lim, up_lim)
                if (0<=Roll_value<=func(*Roll)):
	            break
            output_list.append(Roll)
    return output_list

# Generates a random point in space for a given distance from the point.
# Output: a tuple of coordinates.
# -----
# r: the distance from the origin.
# dim: dimensionality of space.
# r0: the origin coordinate tuple.
def coord_from_r (r, dim):
    random.seed()
    phi = random.uniform (0, 2*math.pi)
    sin_theta = random.uniform (-1, 1)
    if (dim==1):
        coord = np.zeros(1)
        coord[0]=r 
    elif (dim==2):
        coord = np.zeros(2)
        coord[0] = r*math.cos(phi)
        coord[1] = r*math.sin(phi)
    elif (dim==3):
        coord = np.zeros(3)
        coord[0] = r*math.cos(phi)*math.sqrt(1-sin_theta**2)
        coord[1] = r*math.sin(phi)*math.sqrt(1-sin_theta**2)
        coord[2] = r*sin_theta
    else:
        dim = -1
    if (dim==-1): return 0
    else: return coord
    
