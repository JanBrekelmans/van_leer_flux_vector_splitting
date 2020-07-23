# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:25:49 2020

@author: jjwcb
"""

import numpy as np

gamma = 7/5

def f_T(p,state):
    rho_L = state[0]
    u_L = state[1]
    p_L = state[2]
    
    c_L = (gamma*p_L/rho_L)**0.5
    
    AL = 2/((gamma+1)*rho_L)
    BL = (gamma-1)/(gamma+1)*p_L
    
    if p > p_L:
        val = (p-p_L)*(AL/(p+BL))**0.5
    else:
        val = 2*c_L/(gamma-1)*((p/p_L)**((gamma-1)/2/gamma) - 1)
    
    return val

def d_f_T(p,state):
    rho_L = state[0]
    u_L = state[1]
    p_L = state[2]
    
    c_L = (gamma*p_L/rho_L)**0.5
    
    AL = 2/((gamma+1)*rho_L)
    BL = (gamma-1)/(gamma+1)*p_L
    
    if p > p_L:
        val = (AL/(BL+p))**0.5*(1-(p-p_L)/(2*(BL+p)))
    else:
        val = 1/(rho_L*c_L)*(p/p_L)**(-(gamma+1)/(2*gamma))
    
    return val

def p_f(p,left_state,right_state):
    
    du = right_state[1] - left_state[1]
    
    return f_T(p,left_state) + f_T(p,right_state) + du

def d_p_f(p,left_state,right_state):
    
    return d_f_T(p,left_state) + d_f_T(p,right_state)

def calc_pressure(left_state,right_state,tol):
    
    p = 0.5*(left_state[2] + right_state[2])
    p = 0.002
    
    p_old = p + 1
    
    cha = np.abs(p-p_old)/(0.5*(p+p_old))
    
    while cha > tol:
        p_old = p
        
        p = p - p_f(p,left_state,right_state)/d_p_f(p,left_state,right_state)
        
        cha = np.abs(p-p_old)/(0.5*(p+p_old))
        
    return p

def check_vacuum(left_state,right_state):
    rho_L = left_state[0]
    u_L = left_state[1]
    p_L = left_state[2]
    
    c_L = (gamma*p_L/rho_L)**0.5
    
    rho_R = right_state[0]
    u_R = right_state[1]
    p_R = right_state[2]
    
    c_R = (gamma*p_L/rho_L)**0.5
    
    if 2*(c_L+c_R)/(gamma-1) <= u_R-u_L:
        raise Exception("Vacuum detected, aborting computation")

class Riemann_Solution:
    
    tol = 1e-6
    
    def __init__(self,left_state,right_state):
        self.left_state = left_state
        self.right_state = right_state
        
        check_vacuum(left_state, right_state)
        
        self.compute_p_u()
        
        self.determine_structure()
        
        self.print_structure()
        
        self.determine_states()
        
    def compute_p_u(self):
        self.p_star = calc_pressure(self.left_state,self.right_state,self.tol)
        
        self.u_star = 0.5*(self.left_state[1]+self.right_state[1]) + \
            0.5*(f_T(self.p_star,self.right_state) - f_T(self.p_star,self.left_state))
    
    def determine_structure(self):
        if self.p_star > self.left_state[2]:
            self.left_structure = "Shock"
        else:
            self.left_structure = "Rarefaction"
        
        if self.p_star > self.right_state[2]:
            self.right_structure = "Shock"
        else:
            self.right_structure = "Rarefaction"
    
    def print_structure(self):
        print(self.left_structure + " - Contact discontinuity - " + self.right_structure)
    
    def determine_states(self):
        # Left structure
        rho_L = self.left_state[0]
        u_L = self.left_state[1]
        p_L = self.left_state[2]
        AL = 2/((gamma+1)*rho_L)
        BL = (gamma-1)/(gamma+1)*p_L
        p_star = self.p_star
        u_star = self.u_star
        
        if self.left_structure == "Shock":
            QL = ((p_star + BL)/AL)**0.5
            
            temp1 = p_star/p_L + (gamma-1)/(gamma+1)
            temp2 = (gamma-1)*p_star/((gamma+1)*p_L) + 1
            
            self.rho_star_L = rho_L*temp1/temp2
            self.SL = u_L - QL/rho_L
        else:
            self.rho_star_L = rho_L*(p_star/p_L)**(1/gamma)
            c_L = (gamma*p_L/rho_L)**0.5
            a_star_L = c_L*(p_star/p_L)**((gamma-1)/2/gamma)
            self.SHL = u_L - c_L
            self.STL = u_star - a_star_L
        
        # Right structure
        rho_R = self.right_state[0]
        u_R = self.right_state[1]
        p_R = self.right_state[2]
        AR = 2/((gamma+1)*rho_R)
        BR = (gamma-1)/(gamma+1)*p_R
        
        if self.right_structure == "Shock":
            QR = ((p_star + BR)/AR)**0.5
            
            temp1 = p_star/p_R + (gamma-1)/(gamma+1)
            temp2 = (gamma-1)*p_star/((gamma+1)*p_R) + 1
            
            self.rho_star_R = rho_R*temp1/temp2
            self.SR = u_R + QR/rho_R
        else:
            self.rho_star_R = rho_R*(p_star/p_R)**(1/gamma)
            c_R = (gamma*p_R/rho_R)**0.5
            c_star_R = c_R*(p_star/p_R)**((gamma-1)/2/gamma)
            self.SHR = u_R + c_R
            self.STR = u_star + c_star_R
            

    def solution(self,x,x0,time):
        rho_L = self.left_state[0]
        u_L = self.left_state[1]
        p_L = self.left_state[2]
        rho_R = self.right_state[0]
        u_R = self.right_state[1]
        p_R = self.right_state[2]
        
        rho_star_L = self.rho_star_L
        rho_star_R = self.rho_star_R
        u_star = self.u_star
        p_star = self.p_star
        
        n = x.size
        y = np.zeros((3,n))
        
        # Left of the contact discontinuity
        if self.left_structure == "Shock":
            SL = self.SL
            
            for i in range(n):
                if x[i]-x0 <= SL*time:
                    y[0,i] = rho_L
                    y[1,i] = u_L
                    y[2,i] = p_L
                elif x[i] - x0 <= u_star*time:
                    y[0,i] = rho_star_L
                    y[1,i] = u_star
                    y[2,i] = p_star
                else:
                    nr = i
                    break
        else:
            SHL = self.SHL
            STL = self.STL
            c_L = (gamma*p_L/rho_L)**0.5
            
            for i in range(n):
                if x[i] - x0 <= SHL*time:
                    y[0,i] = rho_L
                    y[1,i] = u_L
                    y[2,i] = p_L
                elif x[i] - x0 <= STL*time:
                    xt = (x[i]-x0)/time
                    temp = 2/(gamma+1) + (gamma-1)/((gamma+1)*c_L)*(u_L-xt)
                    y[0,i] = rho_L*temp**(2/(gamma-1))
                    temp = c_L + (gamma-1)/2*u_L + xt
                    y[1,i] = 2*temp/(gamma+1)
                    temp = 2/(gamma+1) + (gamma-1)/((gamma+1)*c_L)*(u_L-xt)
                    y[2,i] = p_L*temp**(2*gamma/(gamma-1))
                elif x[i] - x0 <= u_star*time:
                    y[0,i] = rho_star_L
                    y[1,i] = u_star
                    y[2,i] = p_star
                else:
                    nr = i
                    break
        
        # Right of the discontinuity
        if self.right_structure == "Shock":
            SR = self.SR
            
            for i in range(n-nr):
                i = i + nr
                if x[i]-x0 <= SR*time:
                    y[0,i] = rho_star_R
                    y[1,i] = u_star
                    y[2,i] = p_star
                else:
                    y[0,i] = rho_R
                    y[1,i] = u_R
                    y[2,i] = p_R
        else:
            SHR = self.SHR
            STR = self.STR
            c_R = (gamma*p_R/rho_R)**0.5
            
            for i in range(n-nr):
                i = i + nr
                
                if x[i]-x0 <= STR*time:
                    y[0,i] = rho_star_R
                    y[1,i] = u_star
                    y[2,i] = p_star
                elif x[i]-x0 <= SHR*time:
                    xt = (x[i]-x0)/time
                    temp = 2/(gamma+1) - (gamma-1)/((gamma+1)*c_R)*(u_R-xt)
                    y[0,i] = rho_R*temp**(2/(gamma-1))
                    temp = -c_R + (gamma-1)/2*u_R + xt
                    y[1,i] = 2*temp/(gamma+1)
                    temp = 2/(gamma+1) - (gamma-1)/((gamma+1)*c_R)*(u_R-xt)
                    y[2,i] = p_R*temp**(2*gamma/(gamma-1))
                else:
                    y[0,i] = rho_R
                    y[1,i] = u_R
                    y[2,i] = p_R
                    
            
        return y

if __name__ == "__main__":
    left_state = [1,0,1]
    right_state = [0.125,0,0.1]
    
    sol = Riemann_Solution(left_state,right_state)
    
    