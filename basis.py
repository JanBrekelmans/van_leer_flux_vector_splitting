# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 01:46:44 2020

@author: jjwcb
"""

import os
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

gamma = 7/5
eps = 1e-4

def extend_x(x,n_ex):
    n = x.size
    dx = x[1]-x[0]
    
    
    x_n = np.zeros(n + 2*n_ex)
    x_n[n_ex:-n_ex] = x
    
    for i in range(n_ex):
        x_n[n_ex-1-i] = x[0] - (i+1) * dx
        x_n[-n_ex + i] = x[-1] + (i+1) * dx
        
    return x_n

def time_step(state):
    n = state.shape[1]
    
    dt = 0
    
    for i in range(n):
        rho = state[0,i]
        u = state[1,i]/rho
        e = state[2,i]/rho
        p = rho*(gamma-1)*(e - 0.5*u**2)
        
        c = np.sqrt(gamma*p/rho)
        
        dt = max(dt,np.abs(u)+np.abs(c))
    
    return dt


def save_plot(save_location, iteration, time, x, state, sol, n_ex, plot_limits):
    file_name = "fig{:04d}.jpg".format(iteration)
    path = os.path.join(save_location,file_name)
    
    rho = state[0,n_ex:-n_ex]
    u = state[1,n_ex:-n_ex]/rho
    e = state[2,n_ex:-n_ex]/rho
    e_int = e - 0.5*u**2
    p = rho*(gamma-1)*(e - 0.5*u**2)
    
    x_p = x[n_ex:-n_ex]
    sol_p = sol[:,n_ex:-n_ex]
    e_sol = sol_p[2,:]/((gamma-1)*sol_p[0,:])
    
    plt.clf()
    plt.suptitle("Time {:0.4f}".format(time))
    
    plt.subplot(221)
    plt.plot(x_p,sol_p[0,:],'r')
    plt.plot(x_p,rho,'k')
    plt.title("Density")
    plt.ylim(plot_limits[0])
    
    plt.subplot(222)
    plt.plot(x_p,sol_p[1,:],'r')
    plt.plot(x_p,u,'k')
    plt.title("Velocity")
    plt.ylim(plot_limits[1])
    
    plt.subplot(223)
    plt.plot(x_p,sol_p[2,:],'r')
    plt.plot(x_p,p,'k')
    plt.title("Pressure")
    plt.ylim(plot_limits[2])
    
    plt.subplot(224)
    plt.plot(x_p,e_sol,'r')
    plt.plot(x_p,e_int,'k')
    plt.title("Internal energy")
    plt.ylim(plot_limits[3])
    
    plt.tight_layout()
    
    plt.savefig(path,dpi = 200)

def fix_cavitation(state):
    n = state.shape[1]
    
    rho = state[0,:]
    u = state[1,:]/rho
    e = state[2,:]/rho
    p = (gamma-1)*rho*(e-0.5*u**2)
    
    for i in range(n):
        rho_n = rho[i]
        u_n = u[i]
        p_n = p[i]
        
        if rho[i] <= 0 or np.isnan(rho[i]):
            rho_n = eps
        
        if p[i] <= 0 or np.isnan(p[i]):
            p_n = eps
            
        state[0,i] = rho_n
        # state[1,i] = u_n*rho_n
        state[2,i] = rho_n*(p_n/((gamma-1)*rho_n) + .5*u_n**2)
    
    return state
    
if __name__ == "__main__":
    n = 5
    n_ex = 2
    
    x = np.linspace(0,1,n)
    
    x_n = extend_x(x,n_ex)
    