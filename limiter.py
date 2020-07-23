# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 01:21:54 2020

@author: jjwcb
"""

import numpy as np

"""
Linear limiters
"""
def upwind(r):
    return np.zeros_like(r)

def lax_wendroff(r):
    return np.ones_like(r)

def beam_warming(r):
    return r

def fromm(r):
    return .5*(1+r)

def k(r):
    return 1/3 + 2*r/3


"""
Nonlinear limiters
"""

def minmod(r):
    n = r.size
    
    phi = np.zeros(n)
    
    for i in range(n):
        if r[i] <= 0:
            phi[i] = 0
        elif r[i] <= 1:
            phi[i] = r[i]
        else:
            phi[i] = 1
    
    return phi

def superbee(r):
    n = r.size
    
    phi = np.zeros(n)
    
    for i in range(n):
        if r[i] <= 0:
            phi[i] = 0
        elif r[i] <= .5:
            phi[i] = 2*r[i]
        elif r[i] <= 1:
            phi[i] = 1
        elif r[i] <= 2:
            phi[i] = r[i]
        else:
            phi[i] = 2
    
    return phi

def koren(r):
    n = r.size
    
    phi = np.zeros(n)
    
    for i in range(n):
        if r[i] <= 0:
            phi[i] = 0
        elif r[i] <= .25:
            phi[i] = 2*r[i]
        elif r[i] <= 2.5:
            phi[i] = 1/3 + 2*r[i]/3
        else:
            phi[i] = 2
    
    return phi

def van_leer(r):
    n = r.size
    
    phi = np.zeros(n)
    
    phi = (r+np.abs(r))/(1+np.abs(r))
    
    return phi

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    r = np.linspace(-1,3,50)
    
    y = koren(r)
    
    
    plt.plot(r,y)
    