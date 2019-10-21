from __future__ import division
import numpy as np
import math
import random

def sech (x):
    return 1/np.cosh(x)

def exp_sech (r, z, R=8, Z=1):
    if r>=0: return math.exp(-(r/R))*sech(z/Z)
    else: return 0

def NSF (r, R=50):
    if r>0: return 1/((r/R)*(1+r/R)**2)
    else: return 0

def Einasto (r, R=50, a=0.5):
    if r>0: return math.exp(-(r/R)**a)
    else: return 0

def Hernquist (r, R=30):
    if r>0: return 1/((r/R)*(1+r/R)**3)
    else: return 0

def Jaffe (r, R=10):
    if r>0: return 1/((r/R)**2*(1+r/R)**2)
    

