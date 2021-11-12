#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:14:15 2021

@author: magelineduquesne
"""
# IMPORT MODULES
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from scipy import optimize
import scipy.misc
from scipy.misc import derivative

#os.chdir("C:/Users/marie/Documents/WREProj")

# mean monthly temperature [C]
temperature= pd.read_csv("/Users/magelineduquesne/Documents/Documents/EPFL/SIE/Master_1/WRE/WRE_project/temperature.txt")
T_m=temperature.T.to_numpy()[0]

# hourly precipitation intensity [mm/h] for the period 01/01/2000 to 31/12/2005
precipitation=pd.read_csv("/Users/magelineduquesne/Documents/Documents/EPFL/SIE/Master_1/WRE/WRE_project/P.txt").T.to_numpy()[0]

# Changes in monthly temperature [degrees C]
temperature_change= pd.read_csv("/Users/magelineduquesne/Documents/Documents/EPFL/SIE/Master_1/WRE/WRE_project/temperature_change.txt")
T_c=temperature_change.T.to_numpy()[0]

# monthly mean crop coefficient [-] (average among all the crops and soil uses of the basin
cropcoeff=pd.read_csv("/Users/magelineduquesne/Documents/Documents/EPFL/SIE/Master_1/WRE/WRE_project/kc.txt")
K_c=cropcoeff.T.to_numpy()[0]

# instantaneous discharge at hourly time step [m3/s] for the period 01/01/2000 to 31/12/2004
Q_obs=pd.read_csv("/Users/magelineduquesne/Documents/Documents/EPFL/SIE/Master_1/WRE/WRE_project/Q_obs.txt").T.to_numpy()[0]

plt.plot(precipitation)


# PARAMETERS
s_w=0.25    # [-] Wilting point
s_1=0.4     # [-] soil moisture above which plants transpire at kc*ET0
n=0.3      # [-] Porosity
Q_b=7       # [m3/s] Base flow
t_sup=22    # [h] superficial residence time
A=4000*1e6     # [m²] area of the basin
phi=38     # [degrees] latitude of the basin


# these are the 'free parameters' : they will be determined during next week (session 2 of the project)
# here is a proposed average value that is the right order of magnitude

K_sat=1e-5           # [m/s] Saturated hydraulic conductivity
K_sat_h = K_sat*3600  # [m/h] Saturated hydraulic conductivity
c=10                # [-] exponent of ksat for the equation k = ksat * s^c
t_sub=200            # [h] mean sub-superficial residence time
z=1                  # [m] root zone thickness

day_month=[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] #"day_month": number of days for each month
month_end=np.cumsum(day_month)-1                    #"month_end": last day of each month
month_start=month_end-day_month+1


#Thornthwaite equation
lat = phi                     #latitude of the site (degrees)
D = [k for k in range(365)]   #day of the year   
delta = [0.409*np.sin(2*np.pi*d/365-1.39)for d in D]          # [-]
omega_s = [np.arccos(-np.tan(lat*np.pi/180)*np.tan(i)) for i in delta]   # [rad]
N_D = [24*o/np.pi for o in omega_s]                   # [h] number of daylight hours of day d

N_m = [np.mean(N_D[month_start[m]:month_end[m]]) for m in range(12)]    # [h]  mean daylight hours of month m
Ii = np.sum([np.power(T_m[i]/5, 1.514) for i in range(12)])             # heat index [-]    
a = 6.75e-7 * Ii**3 - 7.71e-5 * Ii**2 + 1.79e-2 * Ii + 0.49           # experimental exponent [-]

# monthly average potential evapotranspiration :
ET_0 = [16*N_m[i] / 12 * (10*T_m[i]/Ii)**a / (24*day_month[i]) for i in range(12)]  # [mm/h]


def month(t):
    """
    Inputs :
        - t a time in hours, given from January 1st as a reference 
    
    Ouptuts :
        - the number of the month the given hour is in.
        Month 0 corresponds to January, and month 11 is December.
    
    month(t) calculates the month in which the given hour is in, regardless if the year.
    The hour 0 must ALWAYS correspond to a January 1st in some year.
    """
    
    j= (t//24)%365 # jour de l'année
    if j >= month_start[11]:
        return 11
    else:
        m=0
        while j >= month_start[m]:
            m=m+1
        return m-1

########################################

def f_ET(t,s):
    """
    Input : 
        - hour [h] at which the computation is done
        - s [-] soil moisture at the given time
    
    Output : 
        - ET [mm/h] the evapotranspiration
    
    Works with the following model of evapotranspiration :
        if s < s_w (wilting point), ET = 0
        if s > s_1 (too much water), ET = ET_0 * K_c
        else : linear interpolation between 0 and ET_0 * K_c
    """
    m = month(t)
    if s <= s_w:
        return 0
    elif (s > s_w and s < s_1):
        return ET_0[m] * K_c[m] / (s_1-s_w) * (s-s_w)
    else:
        return ET_0[m] * K_c[m]
    

################################

def P(t):
    """
    Input : t [h] the hour at which the calculation is done
    Output : P [m3/s] Total precipitation over the whole basin
    
    P(t) = p * Area
    
    /!\ p is given in mm/h
    """
    return precipitation[int(t)]*A*1e-3 / 3600

f_ET(0,0.3)

def hydr_model(K_sat, c, t_sub, z, P, K_c, n_years, s_0 = 0, V_sup_0 = 0, V_sub_0 = 0):
    """
    Inputs :
        - K_sat [m/s] is the saturated hydraulic conductivity (free parameter)
        - c [-] is the exponent of the hydraulic conductivity law (K = K_sat * s**c) (free parameter)
        - t_sub [h] is the mean sub-superficial residence time (free parameter)
        - z [m] is the root zone thickness (free parameter)
        - P [mm/h] is the hourly precipitation, is a vector !
        - K_c [-] is the crop coefficient representative of the whole area
        - n_years [years] is the number of years to process 
    Optional :
        - s_0 [-] soil moisture at time t=0. Defaults to 0.
        - V_sup_0 [m3] Superficial volume of water at time t=0. Defaults to 0.
        - V_sub_0 [m3] Sub-superficial volume of water at time t=0. Defaults to 0.
    
    Output :
        - Q [m3/s] the total discharge 
        - R [mm/h] the runoff         -> possible to change unit if not convenient
        - I [mm/h] the infiltration   -> possible to change unit if not convenient
        - s [-] the soil saturation
        - L [mm/h] the leaching       -> possible to change unit if not convenient
        - ET [mm/h] the actual evapotranspiration
    
    This hydrological model uses a lot of other functions to compute the associated discharges.
    It works with a time step of one hour and does all the computations according to this time step.
    
    This function also uses the following parameters (defined in the beginning of the notebook) :
        - s_w [-] Wilting point
        - s_1 [-] soil moisture threshold
        - n [-] soil porosity
        - Q_b [m3/s] base flow
        - t_sup [h] the average superficial residence time
        - A [m2] the area of the basin
        - phi [degrees] the latitude of the basin    
        
        If you feel anything is missing in this description, please add it !
    """
    
    # Verify that the inputs are correct
    #assert type(n_years)=='int', "the number of years is not an integer"
    # ...
    
    # initalize the output vectors
    n_steps = n_years * 365 * 24
    Q = [0 for i in range(n_steps)]
    R = Q.copy()
    I = Q.copy()
    s = Q.copy()       
    L = Q.copy()
    ET = Q.copy()
    
    # and also :
    q_sup = Q.copy()
    q_sub = Q.copy()
    
    # initializing s, q_sup, q_sub & Q
    s[0] = s_0
    q_sup[0] = V_sup_0 / t_sup
    q_sub[0] = V_sub_0 / t_sub
    Q[0] = A*(q_sup[0] + q_sub[0]) + Q_b
    
    # for each time step do 
    for t in range(n_steps):           
        
        # Infiltration
        I[t] = min(P[t], K_sat*1000*3600)   # [mm/h]
        
        # Runoff
        R[t] = P[t] - I[t]            # [mm/h]
        
        # Evapotranspiration
        ET[t] = f_ET(t, s[t])        #[mm/h]
        
        # Leaching
        L[t] = K_sat * s[t]**c       # [mm/h]
        
        # euler integration :
        dt = 1    # [h]
        
        # soil moisture
        try :
            s[t+1] = s[t] + dt * (I[t]-ET[t]-L[t])/(n*z) /1000  # 1000 factor to account for mm/h converted to m/h
        except IndexError:
            break
        
        # q_sub & q_sup
        q_sub[t+1] = q_sub[t] + dt/t_sub * (R[t] - q_sub[t])    #[mm/h]
        q_sup[t+1] = q_sup[t] + dt/t_sup * (L[t] - q_sup[t])    #[mm/h]
        
        # Q
        Q[t+1] = A * (q_sup[t+1] + q_sub[t+1])/1000/3600 + Q_b
    
    return [Q, R, I, s, L, ET]

print(1e-6*1000*3600)
plt.plot(precipitation[0:int(len(precipitation)/5)])

output = hydr_model(1e-7, c, t_sub, z, precipitation, K_c, 1)


for i in range(6):
    plt.figure()
    plt.plot(output[i])
    plt.show()
    
Q_b
time = [k for k in range(52560)]
sum([P(t) for t in time])
plt.plot(time,[Q_obs[t] for t in time])
#plt.plot(time,[R(t) for t in time])

s=np.linspace(0,1,90)
E=[f_ET(24*90,si) for si in s]
plt.plot(s,E)
plt.title ("Evapotranspitation")
plt.xlabel("soil moisture")
plt.ylabel("ET(s)")



## Part 2 
gamma=9806*10**(-9)  #[N/mm^3]
eta=0.8     #turbine efficienct [-]
delta_z=100*10**(3) #[mm]

def P(Q): 
    return (eta*gamma*delta_z*Q)-(eta*gamma*K_sat*10**(3)*(Q**3))   #
print(scipy.misc.derivative(P,0))
Q_obs_pred=scipy.misc.derivative(P,0)

N_inter = 52560
K_sat_MC=np.zeros(N_inter)
c_MC=np.zeros(N_inter)
t_sub_MC=np.zeros(N_inter)
z_MC=np.zeros(N_inter)
NS=np.zeros(N_inter)

K_sat_MC[0]=K_sat
c_MC[0]=c
t_sub_MC[0]=t_sub
z_MC[0]=z

s_K_sat=np.std(K_sat_MC)
s_c=np.std(c_MC)
s_t_sub=np.std(t_sub_MC)
s_z=np.std(z_MC)

#Q_mod=
#for t in range (52560):
    #NS=1-(sum((Q_obs(t)-Q_mod)^2)/(sum(Q_obs(t)-Q_obs_pred)^2))



