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
import warnings
import random as rd
warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=DeprecationWarning)
# import scipy as scipy
# from scipy import optimize
# import scipy.misc
# from scipy.misc import derivative
# import scipy.stats as stats

dirname = os.path.dirname(__file__)

# meanget monthly temperature [C]
temperature= pd.read_csv(dirname+"/temperature.txt")
T_m=temperature.T.to_numpy()[0]


# hourly precipitation intensity [mm/h] for the period 01/01/2000 to 31/12/2005
precipitation=pd.read_csv(dirname+"/P.txt").T.to_numpy()[0]


# Changes in monthly temperature [degrees C]
temperature_change= pd.read_csv(dirname+"/temperature_change.txt")
T_c=temperature_change.T.to_numpy()[0]


# monthly mean crop coefficient [-] (average among all the crops and soil uses of the basin
cropcoeff=pd.read_csv(dirname+"/kc.txt")
K_c=cropcoeff.T.to_numpy()[0]


# instantaneous discharge at hourly time step [m3/s] for the period 01/01/2000 to 31/12/2004
Q_obs=pd.read_csv(dirname+"/Q_obs.txt").T.to_numpy()[0]


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
z=1000               # [mm] root zone thickness

day_month=[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] #"day_month": number of days for each month
month_end=np.cumsum(day_month)-1                    #"month_end": last day of each month
month_start=month_end-day_month+1


#Thornthwaite equation
lat = phi                     #latitude of the site (degrees)
D = [k for k in range(365)]   #day of the year   
delta = [0.409*np.sin(2*np.pi*d/365-1.39) for d in D]          # [-]
omega_s = [np.arccos(-np.tan(lat*np.pi/180)*np.tan(i)) for i in delta]   # [rad]
N_D = [24*o/np.pi for o in omega_s]                   # [h] number of daylight hours of day d

N_m = [np.mean(N_D[month_start[m]:month_end[m]]) for m in range(12)]    # [h]  mean daylight hours of month m
Ii = np.sum([np.power(T_m[i]/5, 1.514) for i in range(12)])             # heat index [-]    
a = 6.75e-7 * Ii**3 - 7.71e-5 * Ii**2 + 1.79e-2 * Ii + 0.49           # experimental exponent [-]

# monthly average potential evapotranspiration :
ET_0 = [16*N_m[i] / 12 * (10*T_m[i]/Ii)**a / (24*day_month[i]) for i in range(12)]  # [mm/h]


##########################################"

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

def downscaling(Pdaily):
    """
    Downscale daily precipitation to hourly precipitation assuming that hourly
    rainfall are exponentially distributed.

    INPUT
    "Pdaily": precipitation at daily timestep

    OUTPUT
    "Phourly": precipitation at hourly time step 

    if "Pdaily" is in [mm/day], "Phourly" is in [mm/h]     

    """

    Phourly = np.empty((len(Pdaily)*24,))
    for i in range(len(Pdaily)):
        #print(i)
        distr = -np.log(np.random.rand(24,))
        sumdistr = np.sum(distr)
        Phourly[i*24:(i+1)*24] = Pdaily[i]*distr/sumdistr

    return Phourly    


################################



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
    
    # Verify that the inputs' length are correct
    if len(P) != 365 * 24 * n_years:
        print("Warning !The precipitation file is too long for the number of years indicated.\
              Truncating the file to the correct length")
        P = P[:365*24*n_years].copy()
    
    # count the number of errors
    error_count = 0
    
    # initalize the output vectors
    n_steps = n_years * 365 * 24
    Q = [0 for i in range(n_steps)]
    R = Q.copy()
    I = Q.copy()
    s = Q.copy()       
    L = Q.copy()
    ET = Q.copy()
    
    # and also :
    global q_sup
    global q_sub
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
        # parfois le programme s'arrête, la suite est pour stopper proprement
        # le programme a ce moment
        try:
            L[t] = K_sat * s[t]**c       # [mm/h]
        except:
            print("program has a problem with  L[t] = K_sat * s[t]**c       # [mm/h]")
            print("L[t] = ", L[t])
            print("K_sat = ", K_sat)
            print("s[t] = ", s[t])
            print("c = ", c)
            error_count += 1
        
        # euler integration :
        dt = 1    # [h]
        
        # soil moisture
        try :
            s[t+1] = s[t] + dt * (I[t]-ET[t]-L[t])/(n*z)
            
            if s[t+1] < 0:
                # print("\nWARNING !")
                # print("    Soil moisture negative (value = "+ str(s[t+1]) + ") for time t="+ str(t+1))
                # ans = input("Ignore and set value to 0 ? [y] / [n] ")
                # if ans == "y":
                #     print("    Setting value to 0\n")
                # elif ans == "n":
                #     print("Aborting...")
                #     raise ValueError("The value of the soil moisture is negative !")
                s[t+1] = 0
                error_count += 1
            elif s[t+1] > 1:
                s[t+1] = 1
                error_count += 1
        except IndexError:
            break
        
        # q_sub & q_sup
        q_sub[t+1] = q_sub[t] + dt/t_sub * (R[t] - q_sub[t])    #[mm/h]
        q_sup[t+1] = q_sup[t] + dt/t_sup * (L[t] - q_sup[t])    #[mm/h]
        
        # Q
        Q[t+1] = A * (q_sup[t+1] + q_sub[t+1])/1000/3600 + Q_b
    
    return [Q, R, I, s, L, ET]


##################################################



def plot_model(K_sat, c, t_sub, z, P, K_c, n_years, s_0 = 0, V_sup_0 = 0, V_sub_0 = 0):
    """
    plots the outputs of the hydrological model using the same parameters.
    
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
    
    Output (plots) :
        - Q [m3/s] the total discharge 
        - R [mm/h] the runoff         -> possible to change unit if not convenient
        - I [mm/h] the infiltration   -> possible to change unit if not convenient
        - s [-] the soil saturation
        - L [mm/h] the leaching       -> possible to change unit if not convenient
        - ET [mm/h] the actual evapotranspiration
    """

    out = hydr_model(K_sat, c, t_sub, z, P, K_c, n_years, s_0, V_sup_0, V_sub_0)
    
    lines = 2
    col = 3
    
    fig, axs = plt.subplots(lines, col)
    titres = ["Discharge [m3/s]", "R [mm/h]", "I [mm/h]", "s [-]", "L [mm/h]", "ET [mm/h]"]
    
    for i in range(lines):
        for j in range(col):
            axs[i, j].plot(out[j+3*i])
            axs[i, j].set_title(titres[j+3*i])
    
    plt.show()
    return None



##############################################♀

def check_model(K_sat, c, t_sub, z, P, K_c, n_years):
    out = hydr_model(K_sat, c, t_sub, z, P, K_c, n_years)
    
    s = out[3]
    P_tot = np.sum(P)       # mm/h
    R_tot = np.sum(out[1])
    L_tot = np.sum(out[4])
    ET_tot = np.sum(out[5])
    q_sup_tot = np.sum(q_sup)
    q_sub_tot = np.sum(q_sub)
    
    testS = P_tot / (ET_tot + R_tot + L_tot + n*z*(s[-1]-s[0]))
    testQ = (P_tot - ET_tot) / (q_sub_tot + q_sup_tot + n*z*(s[-1]-s[0]) + q_sup[-1]*t_sup + q_sub[-1]*t_sub)
    
    return testS, testQ
    
    
#########################################



# Define new parameters for the parameter optimization


Q_avg = [np.mean(Q_obs) for i in Q_obs]

theta_absolute_max = [5e-6, 10, 200, 1000]
ns_absolute_max = float('-inf')


def T_SA(i):
    """
    fonction qui calcule la 'temperature liee a l'algorithme pour savoir si on
    va voir ailleurs
    """
    c_r = 1/1200        # cooling rate
    return np.exp(-c_r*i)

def NS(Q):
    """
    indicateur de proximité du Q en entree avec le Q observe (en donnee de l'exo)
    """
    # print(Q, Q_obs, Q_mod_avg)
    # print(np.diff(Q, Q_obs))
    # print(np.diff(Q, Q_mod_avg))
    # print(Q)
    global Q_obs
    global Q_avg
    
    a = np.sum(  np.power(  np.subtract(Q, Q_obs) , 2)   )
    b = np.sum(  np.power(  np.subtract(Q, Q_avg) , 2)  )
    
    # print("    a = ", a)
    # print("    b = ", b)
    
    try:
        ans = 1 - a/b
    except:
        ans = float("-inf")
    return ans


def opt_param(theta_start = [5e-6, 10, 200, 1000]):
    
    global Q_obs
    global theta_absolute_max
    global ns_absolute_max
    
    theta_old = theta_start.copy()  # initial values of the parameters
    theta_new = theta_old.copy()
    theta_minmax = [[1e-7, 1e-5], 
                    [1, 20],
                    [1, 400],
                    [1, 2000]]      # min/max values of the parameters
    theta_var = [np.diff(i)[0]/20 for i in theta_minmax]   # variance of the parameter, equal to 5% of the range
    
    
    
    # parameters
    ns_old = float("-inf")      # value of the NS coefficient
    ns_new = 0
    n_sim = 0       # nb of simulations yet
    seuil = 0.87    # seuil pour le NS coeff  
    iteration_max = 1e3
    print("Seuil choisi de : ", seuil)
    
    print("Starting parameters : ", theta_old)
    
    
    while ns_old < seuil:
        n_sim += 1
        # print(n_sim)
        # generate new parameters
        for i in range(4):
            keep_gen = True
            while keep_gen: 
                theta_new[i] = np.random.normal(loc=theta_old[i], scale=theta_var[i])
                
                if theta_new[i] > theta_minmax[i][0] and theta_new[i] < theta_minmax[i][1]:
                    keep_gen = False
               
        Q_mod = hydr_model(theta_new[0], theta_new[1], theta_new[2], theta_new[3], precipitation, K_c, 6)[0]
        ns_new = NS(Q_mod)
        
        # print(ns_new)
        # print("#"*50)
        # print(ns_old)
        
        if ns_new > ns_absolute_max:
            print("\n    NS maximum absolu amélioré (iteration " + str(n_sim) + ")")
            print("    NS_max_absolu = ", ns_new)
            print("    Paramètres : ", np.round(theta_absolute_max, decimals=3))
            theta_absolute_max = theta_new.copy()
            ns_absolute_max = ns_new
        
        if ns_new > ns_old:
            print("\nValeur de NS améliorée (iteration " + str(n_sim) + ")")
            print("NS = ", ns_new)
            theta_old = theta_new.copy()
            ns_old = ns_new
        
        elif np.random.uniform() < np.exp(-abs(ns_new-ns_old)/T_SA(n_sim)):
            print("\nOn va voir ailleurs (iteration " + str(n_sim) + ")")
            print("NS = ", ns_new)
            
            theta_old = theta_new.copy()
            ns_old = ns_new
        
        if n_sim > iteration_max:
            print("Iterations maximales dépassées pour la boucle principale")
            break
        
    return theta_absolute_max


#########################################
#alpha lambda
def parametres():
    precipitationj=[sum(precipitation[24*k:24*k+24]) for k in range(0,6*365) ]
    n_r1=[0]*72
    I_r1=[0]*72
   
    jour=0
    for i in range (0,72):
        n=0
        I=0
        m=144%12
        for k in range (jour,jour+day_month[m]):
            
            if precipitationj[k]!=0:
                n=n+1
                I=I+precipitationj[k]
            n_r1[i]=n
            I_r1[i]=I
        jour=jour+day_month[m]-1
    
    n_rainy=[0]*12
    I_rainy=[0]*12
    for i in range (0,12):
        n=0
        I=0
        for j in range (0,6):
            n=n+n_r1[j*12+i]
            I=I+I_r1[j*12+i]
        n_rainy[i]=n
        I_rainy[i]=I
        
    lambda_=[0]*12
    alpha=[0]*12
    
    for i in range(0,12):
        lambda_[i]=n_rainy[i]/(day_month[i]*6)
        alpha[i]=I_rainy[i]/n_rainy[i]
        
    return (lambda_,alpha)
    
    
    
############################################



def rain_gen(years=100):
    
    lambda_,alpha=parametres()
    
    
    output = [0 for i in range(365*years)]
    total_day = 0
    
    for y in range(years):
        for m in range(12):
            for d in range(day_month[m]):
                
                # Does it rain ?
                if rd.random() < lambda_[m]:
                    output[total_day] = rd.expovariate(1/alpha[m])
                total_day += 1
                
    
    return downscaling(output)
    
#########################################☻
def rain_gen2(years=100):
    
    lambda_,alpha=parametres()
    
    
  
    output = [0 for i in range(365*years)]
    total_day = 0

    
    for y in range(years):
        for m in range(12):
            day_rain=0
            I=0
            for d in range(day_month[m]):
                
                # Does it rain ?
                if rd.random() < lambda_[m]:
                    output[total_day] = rd.expovariate(1/alpha[m])
                    I=I+output[total_day]
                    day_rain=day_rain+1
                total_day += 1
                
    mois=[[]]*12
    for i in range (0,years):
        for m in range (0,12):
            mois[m]=mois[m]+[output[i*365+month_start[m]:i*365+month_end[m]]]
    moyenne=[np.mean(mois[m]) for m in range (0,12)]
    standard= [np.std(mois[m])for m in range(0,12)]
    
    return (downscaling(output),moyenne,standard)



