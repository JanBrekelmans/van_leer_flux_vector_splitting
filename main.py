# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 10:16:25 2020

@author: jjwcb
"""

import numpy as np

import basis
import initialization
import flux
import limiter
import riemann

"""
User-defined variables
"""
# Number of volumes
N = 100

# Extending the grid to incorporate boundary conditions
n_ex = 2

# Which test we want to run
sod_test_num = 1

# Courant Lax Friedrichs constant
CLF = 0.25

# Which flux vector splitting we want to use, flux_p = positive part, flux_n = negative part
flux_p = flux.van_leer_p
flux_n = flux.van_leer_n

# Which limiter function we want to use
phi = limiter.upwind

# Fix cavitation issues?
fix_cavitation = False

# Save figure on every timestep?
save_fig = True
save_location = r".\plots"
plot_limits = initialization.sod_limits(sod_test_num)

"""
Back-end
"""
x = np.linspace(0,1,N+1)
x = 0.5*(x[1:] + x[:-1])
x = basis.extend_x(x, n_ex)
dx = x[1] - x[0]

end_time = initialization.sod_times(sod_test_num)
init = lambda a : initialization.sod_conservative(sod_test_num, a)
init_riemann = initialization.sod_riemann_states(sod_test_num)
plot_limits = initialization.sod_limits(sod_test_num)

# Initialize arrays
state = init(x)
flux1 = np.zeros_like(state)
flux2 = np.zeros_like(state)
flux3 = np.zeros_like(state)
left_state,right_state = init_riemann
riemann_sol = riemann.Riemann_Solution(left_state, right_state)

time = 0
iteration = 0

sol = riemann_sol.solution(x, initialization.sod_split(sod_test_num), time)

if save_fig:
    basis.save_plot(save_location, iteration, time, x, state, sol, n_ex, plot_limits)
    iteration += 1

while time < end_time:
    
    # Calculate time step
    dt = CLF*dx/basis.time_step(state)
    
    if time + dt >= end_time:
        dt = end_time - time
    
    print(time)
    
    # Time integration
    flux1 = -dt*flux.calculate_flux(state, flux_p, flux_n, phi, n_ex,fix_cavitation)/dx
    
    temp = state + flux1
    if fix_cavitation:
        temp = basis.fix_cavitation(temp)
    flux2 = -dt*flux.calculate_flux(temp, flux_p, flux_n, phi, n_ex,fix_cavitation)/dx
    
    temp = state + .25*flux1 + .25*flux2
    if fix_cavitation:
        temp = basis.fix_cavitation(temp)
    flux3 = -dt*flux.calculate_flux(temp, flux_p, flux_n, phi, n_ex,fix_cavitation)/dx
        
    state = state + (flux1 + flux2 + 4*flux3)/6
    
    if fix_cavitation:
        state = basis.fix_cavitation(state)
    
    
    time += dt
    if save_fig:
        sol = riemann_sol.solution(x, initialization.sod_split(sod_test_num), time)
        basis.save_plot(save_location, iteration, time, x, state, sol, n_ex, plot_limits)
        iteration += 1
        
sol = riemann_sol.solution(x, initialization.sod_split(sod_test_num), time)
err = np.sum(np.abs(sol[0,:]-state[0,:])**1/N)**1
print(err)
basis.save_plot(save_location, iteration, time, x, state, sol, n_ex, plot_limits)