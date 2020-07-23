# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 01:38:11 2020

@author: jjwcb
"""

import numpy as np

def sod_riemann_states(num):
    
    if num == 1:
        rho_L = 1.0
        rho_R = 0.125
        u_L = 0
        u_R = 0
        p_L = 1
        p_R = 0.1
    elif num == 2:
        rho_L = 1.0
        rho_R = 1.0
        u_L = -2
        u_R = 2
        p_L = 0.4
        p_R = 0.4
    elif num == 3:
        rho_L = 1.0
        rho_R = 1.0
        u_L = 0
        u_R = 0
        p_L = 1000
        p_R = 0.01
    elif num == 4:
        rho_L = 5.99924
        rho_R = 5.99924
        u_L = 19.5975
        u_R = -6.19633
        p_L = 460.894
        p_R = 46.0950
    elif num == 5:
        rho_L = 1
        rho_R = 1
        u_L = -19.5975
        u_R = -19.5975
        p_L = 1000
        p_R = 0.01
    
    
    left_state = [rho_L,u_L,p_L]
    right_state = [rho_R,u_R,p_R]
    
    return left_state,right_state

def sod_conservative(num,x):
    n = x.size
    
    state = np.zeros((3,n))
    gamma = 7/5
    
    left_state,right_state = sod_riemann_states(num)
    
    rho_L = left_state[0]
    rho_R = right_state[0]
    
    u_L = left_state[1]
    u_R = right_state[1]
    
    rho_u_L = rho_L*u_L
    rho_u_R = rho_L*u_R
    
    p_L = left_state[2]
    p_R = right_state[2]
    
    e_L = p_L/((gamma-1)*rho_L) + 0.5*u_L**2
    e_R = p_R/((gamma-1)*rho_R) + 0.5*u_R**2
    
    rho_e_L = rho_L*e_L
    rho_e_R = rho_R*e_R
    
    x_0 = sod_split(num)
    
    state[0,:] = rho_L*(x <= x_0) + rho_R*(x > x_0)
    state[1,:] = rho_u_L*(x <= x_0) + rho_u_R*(x > x_0)
    state[2,:] = rho_e_L*(x <= x_0) + rho_e_R*(x > x_0)
    
    return state


def sod_limits(num):
    if num == 1:
        return [[-.5,1.5],[-.5,1.5],[-.5,1.5],[1.5,3]]
    elif num == 2:
        return [[-.1,1.1],[-2.5,2.5],[-.1,0.5],[0,1.1]]
    elif num == 3:
        return [[-.1,8],[-2.5,25],[-50,1200],[-50,2700]]
    elif num == 4:
        return [[-1,35],[-10,25],[-50,1800],[0,320]]
    elif num == 5:
        return [[0,9],[-25,5],[-50,1200],[0,2750]]

def sod_times(num):
    if num == 1:
        return 0.25
    elif num == 2:
        return 0.15
    elif num == 3:
        return 0.012
    elif num == 4:
        return 0.035
    elif num == 5:
        return 0.012

def sod_split(num):
    if num == 1:
        return 0.5
    elif num == 2:
        return 0.5
    elif num == 3:
        return 0.5
    elif num == 4:
        return 0.4
    elif num == 5:
        return 0.8
    