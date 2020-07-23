# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:25:49 2020

@author: jjwcb
"""

import numpy as np

gamma = 7/5

def f_T(p,state):
    rhoL = state[0]
    uL = state[1]
    pL = state[2]
    
    aL = (gamma*pL/rhoL)**0.5
    
    AL = 2/((gamma+1)*rhoL)
    BL = (gamma-1)/(gamma+1)*pL
    
    if p > pL:
        val = (p-pL)*(AL/(p+BL))**0.5
    else:
        val = 2*aL/(gamma-1)*((p/pL)**((gamma-1)/2/gamma) - 1)
    
    return val

def d_f_T(p,state):
    rhoL = state[0]
    uL = state[1]
    pL = state[2]
    
    aL = (gamma*pL/rhoL)**0.5
    
    AL = 2/((gamma+1)*rhoL)
    BL = (gamma-1)/(gamma+1)*pL
    
    if p > pL:
        val = (AL/(BL+p))**0.5*(1-(p-pL)/(2*(BL+p)))
    else:
        val = 1/(rhoL*aL)*(p/pL)**(-(gamma+1)/(2*gamma))
    
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
    rhoL = left_state[0]
    uL = left_state[1]
    pL = left_state[2]
    
    aL = (gamma*pL/rhoL)**0.5
    
    rhoR = right_state[0]
    uR = right_state[1]
    pR = right_state[2]
    
    aR = (gamma*pL/rhoL)**0.5
    
    if 2*(aL+aR)/(gamma-1) <= uR-uL:
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
        rhoL = self.left_state[0]
        uL = self.left_state[1]
        pL = self.left_state[2]
        AL = 2/((gamma+1)*rhoL)
        BL = (gamma-1)/(gamma+1)*pL
        p_star = self.p_star
        u_star = self.u_star
        
        if self.left_structure == "Shock":
            QL = ((p_star + BL)/AL)**0.5
            
            temp1 = p_star/pL + (gamma-1)/(gamma+1)
            temp2 = (gamma-1)*p_star/((gamma+1)*pL) + 1
            
            self.rho_starL = rhoL*temp1/temp2
            self.SL = uL - QL/rhoL
        else:
            self.rho_starL = rhoL*(p_star/pL)**(1/gamma)
            aL = (gamma*pL/rhoL)**0.5
            a_starL = aL*(p_star/pL)**((gamma-1)/2/gamma)
            self.SHL = uL - aL
            self.STL = u_star - a_starL
        
        # Right structure
        rhoR = self.right_state[0]
        uR = self.right_state[1]
        pR = self.right_state[2]
        AR = 2/((gamma+1)*rhoR)
        BR = (gamma-1)/(gamma+1)*pR
        
        if self.right_structure == "Shock":
            QR = ((p_star + BR)/AR)**0.5
            
            temp1 = p_star/pR + (gamma-1)/(gamma+1)
            temp2 = (gamma-1)*p_star/((gamma+1)*pR) + 1
            
            self.rho_starR = rhoR*temp1/temp2
            self.SR = uR + QR/rhoR
        else:
            self.rho_starR = rhoR*(p_star/pR)**(1/gamma)
            aR = (gamma*pR/rhoR)**0.5
            a_starR = aR*(p_star/pR)**((gamma-1)/2/gamma)
            self.SHR = uR + aR
            self.STR = u_star + a_starR
            

    def solution(self,x,x0,time):
        rhoL = self.left_state[0]
        uL = self.left_state[1]
        pL = self.left_state[2]
        rhoR = self.right_state[0]
        uR = self.right_state[1]
        pR = self.right_state[2]
        
        rho_starL = self.rho_starL
        rho_starR = self.rho_starR
        u_star = self.u_star
        p_star = self.p_star
        
        n = x.size
        y = np.zeros((3,n))
        
        # Left of the contact discontinuity
        if self.left_structure == "Shock":
            SL = self.SL
            
            for i in range(n):
                if x[i]-x0 <= SL*time:
                    y[0,i] = rhoL
                    y[1,i] = uL
                    y[2,i] = pL
                elif x[i] - x0 <= u_star*time:
                    y[0,i] = rho_starL
                    y[1,i] = u_star
                    y[2,i] = p_star
                else:
                    nr = i
                    break
        else:
            SHL = self.SHL
            STL = self.STL
            aL = (gamma*pL/rhoL)**0.5
            
            for i in range(n):
                if x[i] - x0 <= SHL*time:
                    y[0,i] = rhoL
                    y[1,i] = uL
                    y[2,i] = pL
                elif x[i] - x0 <= STL*time:
                    xt = (x[i]-x0)/time
                    temp = 2/(gamma+1) + (gamma-1)/((gamma+1)*aL)*(uL-xt)
                    y[0,i] = rhoL*temp**(2/(gamma-1))
                    temp = aL + (gamma-1)/2*uL + xt
                    y[1,i] = 2*temp/(gamma+1)
                    temp = 2/(gamma+1) + (gamma-1)/((gamma+1)*aL)*(uL-xt)
                    y[2,i] = pL*temp**(2*gamma/(gamma-1))
                elif x[i] - x0 <= u_star*time:
                    y[0,i] = rho_starL
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
                    y[0,i] = rho_starR
                    y[1,i] = u_star
                    y[2,i] = p_star
                else:
                    y[0,i] = rhoR
                    y[1,i] = uR
                    y[2,i] = pR
        else:
            SHR = self.SHR
            STR = self.STR
            aR = (gamma*pR/rhoR)**0.5
            
            for i in range(n-nr):
                i = i + nr
                
                if x[i]-x0 <= STR*time:
                    y[0,i] = rho_starR
                    y[1,i] = u_star
                    y[2,i] = p_star
                elif x[i]-x0 <= SHR*time:
                    xt = (x[i]-x0)/time
                    temp = 2/(gamma+1) - (gamma-1)/((gamma+1)*aR)*(uR-xt)
                    y[0,i] = rhoR*temp**(2/(gamma-1))
                    temp = -aR + (gamma-1)/2*uR + xt
                    y[1,i] = 2*temp/(gamma+1)
                    temp = 2/(gamma+1) - (gamma-1)/((gamma+1)*aR)*(uR-xt)
                    y[2,i] = pR*temp**(2*gamma/(gamma-1))
                else:
                    y[0,i] = rhoR
                    y[1,i] = uR
                    y[2,i] = pR
                    
            
        return y

if __name__ == "__main__":
    left_state = [1,0,1]
    right_state = [0.125,0,0.1]
    
    sol = Riemann_Solution(left_state,right_state)
    
    