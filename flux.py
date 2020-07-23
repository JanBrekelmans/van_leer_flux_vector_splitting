# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 01:21:53 2020

@author: jjwcb
"""

import numpy as np

gamma = 7/5
eps = 1e-12

def calculate_flux(state, flux_p, flux_n, limiter, n_ex,fix_cavitation):
    flux = np.zeros_like(state)
    N = state.shape[1] - 2*n_ex
    
    for i in range(N):
        # We consider c = i + n_ex cell
        c = i + n_ex
        
        # Left face
        r = (state[:,c] - state[:,c-1] + eps)/(state[:,c-1] - state[:,c-2] + eps)
        temp_state = state[:,c-1] + .5*limiter(r)*(state[:,c-1] - state[:,c-2])
        flux[:,c] -= flux_p(temp_state,fix_cavitation)
        
        r = (state[:,c-1] - state[:,c] + eps)/(state[:,c] - state[:,c+1] + eps)
        temp_state = state[:,c] + .5*limiter(r)*(state[:,c] - state[:,c+1])
        flux[:,c] -= flux_n(temp_state,fix_cavitation)
        
        # Right face
        r = (state[:,c+1] - state[:,c] + eps)/(state[:,c] - state[:,c-1] + eps)
        temp_state = state[:,c] + .5*limiter(r)*(state[:,c] - state[:,c-1])
        flux[:,c] += flux_p(temp_state,fix_cavitation)
        
        r = (state[:,c] - state[:,c+1] + eps)/(state[:,c+1] - state[:,c+2] + eps)
        temp_state = state[:,c+1] + .5*limiter(r)*(state[:,c+1] - state[:,c+2])
        flux[:,c] += flux_n(temp_state,fix_cavitation)
    
    return flux


def van_leer_p(state,fix_cavitation):
    # Calculate values
    rho = state[0]
    u = state[1]/rho
    e = state[2]/rho
    p = rho*(gamma-1)*(e - 0.5*u**2)
    
    if fix_cavitation:
        if np.isnan(rho) or rho <= 1e-4:
            rho = 1e-4
            
        if np.isnan(p) or p <= 1e-4:
            p = 1e-4
    
    c = np.sqrt(gamma*p/rho)
    M = u/c
    # Initiate flux vector
    G = np.zeros(3)
    
    if M <= -1:
        pass
    elif M < 1:
        G[0] = rho*c*((M+1)/2)**2
        G[1] = rho*c**2*(.5*(M+1))**2*((gamma-1)/gamma*M + 2/gamma)
        G[2] = gamma**2/(2*(gamma**2-1))*G[1]**2/G[0]
    else:
        G[0] = rho*c*M
        G[1] = rho*c**2*(M**2 + 1/gamma)
        G[2] = rho*c**3*M*(.5*M**2+1/(gamma-1))
    
    return G

def van_leer_n(state,fix_cavitation):
    # Calculate values
    rho = state[0]
    u = state[1]/rho
    e = state[2]/rho
    p = rho*(gamma-1)*(e - 0.5*u**2)
    
    if fix_cavitation:
        if np.isnan(rho) or rho <= 1e-4:
            rho = 1e-4
            
        if np.isnan(p) or p <= 1e-4:
            p = 1e-4

    c = (gamma*p/rho)**.5
    M = u/c
    
    # Initiate flux vector
    G = np.zeros(3)
    
    if M <= -1:
        G[0] = rho*c*M
        G[1] = rho*c**2*(M**2 + 1/gamma)
        G[2] = rho*c**3*M*(.5*M**2+1/(gamma-1))
    elif M < 1:
        G[0] = -rho*c*((-M+1)/2)**2
        G[1] = rho*c**2*(.5*(-M+1))**2*((gamma-1)/gamma*-M + 2/gamma)
        G[2] = gamma**2/(2*(gamma**2-1))*G[1]**2/G[0]
    else:
        pass
    
    return G