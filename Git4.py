#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:14:15 2021

@author: magelineduquesne
"""
#%% IMPORT MODULES
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


#%% Get files from directory

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

# Area rating curve
# A_rating[0] = area [m2] at lake level 0m
extract=pd.read_csv(dirname+"/area_rating_curve.txt").T.to_numpy()[0]
A_rating = [int(ele[13:]) for ele in extract[1:]]

# changes in monthly mean rainfall intensity of rainy days [%]
alpha_c=pd.read_csv(dirname+"/alpha_change.txt").T.to_numpy()[0]
#temperature in a climate change scenario [degrees C]
T_future=T_m+T_c 

# change in monthly occurence of rainy days [%]
lambda_c=pd.read_csv(dirname+"/lambda_change.txt").T.to_numpy()[0]
#%% set up gobal PARAMETERS

s_w = 0.25    # [-] Wilting point
s_1 = 0.4     # [-] soil moisture above which plants transpire at kc*ET0
n = 0.3      # [-] Porosity
Q_b = 7       # [m3/s] Base flow
t_sup = 22    # [h] superficial residence time
A = 4000*1e6     # [m²] area of the basin
phi = 38     # [degrees] latitude of the basin


# these are the 'free parameters' : they will be determined during next week (session 2 of the project)
# here is a proposed average value that is the right order of magnitude

K_sat = 1e-6          # [m/s] Saturated hydraulic conductivity
K_sat_h = K_sat*3600  # [m/h] Saturated hydraulic conductivity
c = 10                # [-] exponent of ksat for the equation k = ksat * s^c
t_sub = 200            # [h] mean sub-superficial residence time
z = 1000               # [mm] root zone thickness
Qcity = 1              # [m3/s] what does it describe ?
day_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] #"day_month": number of days for each month
month_end = np.cumsum(day_month)-1                    #"month_end": last day of each month
month_start = month_end-day_month+1



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

month_name=["january","february","mars","april","may","june","july","august","september","ocotber","november","december"]



#%% Define the auxiliairy functions

def ET_0(T):
    """
    -INPUT: Monthly temperature
    - OUTPUT : Monthly average potential evapotranspiration in [mm/h]
    
    """
    evap=[16*N_m[i] / 12 * (10*T[i]/Ii)**a / (24*day_month[i]) for i in range(12)]
    return evap

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

def f_ET(t,s,T=T_m):
    """
    Input : 
        - hour [h] at which the computation is done
        - s [-] soil moisture at the given time
        - T [degrees °C] monthly temperature
    
    Output : 
        - ET [mm/h] the evapotranspiration
    
    Works with the following model of evapotranspiration :
        if s < s_w (wilting point), ET = 0
        if s > s_1 (too much water), ET = ET_0 * K_c
        else : linear interpolation between 0 and ET_0 * K_c
    """
    Evap0=ET_0(T)
    m = month(t)
    if s <= s_w:
        return 0
    elif (s > s_w and s < s_1):
        return Evap0[m] * K_c[m] / (s_1-s_w) * (s-s_w)
    else:
        return Evap0[m] * K_c[m]
    
    
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



#%% Hydrological model
#   1. hydr_model models the hydrological quantites according to the parameters
#   2. plot_model plots the differents curves that the model gives out
#   3. check_model checks if the model is correct using the different balance equations

def hydr_model(K_sat, c, t_sub, z, P, K_c, n_years, s_0 = 0, V_sup_0 = 0, V_sub_0 = 0,T=T_m):
    """
    Inputs :
        - K_sat [m/s] is the saturated hydraulic conductivity (free parameter)
        - c [-] is the exponent of the hydraulic conductivity law (K = K_sat * s**c) (free parameter)
        - t_sub [h] is the mean sub-superficial residence time (free parameter)
        - z [mm] is the root zone thickness (free parameter)
        - P [mm/h] is the hourly precipitation, is a vector !
        - K_c [-] is the crop coefficient representative of the whole area
        - n_years [years] is the number of years to process 
    Optional :
        - s_0 [-] soil moisture at time t=0. Defaults to 0.
        - V_sup_0 [m3] Superficial volume of water at time t=0. Defaults to 0.
        - V_sub_0 [m3] Sub-superficial volume of water at time t=0. Defaults to 0.
        - T [degrees °C] monthly temperature, can be changed in a climate change scenario
    
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
        print("Warning ! The precipitation file is too long for the number of years indicated.\
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
        
        if s[t] > 1 :
            s[t] = 1
        elif s[t] < 0:
            s[t] = 0
        
        # Infiltration
        I[t] = min(P[t], K_sat*1000*3600)   # [mm/h]
        
        # Runoff
        R[t] = P[t] - I[t]            # [mm/h]
        
        # Evapotranspiration
        ET[t] = f_ET(t, s[t],T)        #[mm/h]
        
        # Leaching
        # parfois le programme s'arrête, la suite est pour stopper proprement
        # le programme a ce moment
        try:
            L[t] =1000*3600*( K_sat * s[t]**c)     # [mm/h]
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
                # s[t+1] = 0
                error_count += 1
            elif s[t+1] > 1:
                # s[t+1] = 1
                error_count += 0.0001
        except IndexError:
            break
        
        # q_sub & q_sup
        q_sub[t+1] = q_sub[t] + dt/t_sub * (L[t] - q_sub[t])    #[mm/h]
        q_sup[t+1] = q_sup[t] + dt/t_sup * (R[t] - q_sup[t])    #[mm/h]
        
        # Q
        Q[t+1] = A * (q_sup[t+1] + q_sub[t+1])/1000/3600 + Q_b
    
    return [Q, R, I, s, L, ET]


##################################################



def plot_model(Q, R, I, s, L, ET):
    """
    plots the outputs of the hydrological model using the same parameters.
    
    Inputs :
        - Q [m3/s] the total discharge 
        - R [mm/h] the runoff         -> possible to change unit if not convenient
        - I [mm/h] the infiltration   -> possible to change unit if not convenient
        - s [-] the soil saturation
        - L [mm/h] the leaching       -> possible to change unit if not convenient
        - ET [mm/h] the actual evapotranspiration
    
    Output (plots) :
        - Q [m3/s] the total discharge 
        - R [mm/h] the runoff         -> possible to change unit if not convenient
        - I [mm/h] the infiltration   -> possible to change unit if not convenient
        - s [-] the soil saturation
        - L [mm/h] the leaching       -> possible to change unit if not convenient
        - ET [mm/h] the actual evapotranspiration
    """

    out = [Q, R, I, s, L, ET]
    n_years=int(len(Q)/(365*24))
    
    lines = 2
    col = 3
    
    fig, axs = plt.subplots(lines, col,figsize=(18,10))
    titres = ["Discharge [m3/s]", "Runoff [mm/h]", "Infiltration [mm/h]", "soil moisture [-]", "Leaching [mm/h]", "Evaporation [mm/h]"]
    units=["m3/s","mm/h","mm/h","-","mm/h","mm/h"]
    for i in range(lines):
        for j in range(col):
            plt.subplots_adjust(hspace =0.3,wspace=0.2)
            axs[i, j].plot(out[j+3*i])
            axs[i, j].set_title(titres[j+3*i],fontsize=20)
            axs[i, j].set_xlabel("time in hours")
            axs[i, j].set_ylabel(units[j+3*i],fontsize=10)
    
    title=" Time series -  "+str(n_years) + " years"
    plt.suptitle(title,fontsize=25)
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
    





#%% Finding the best parameters



# Define new parameters for the parameter optimization

theta_absolute_max = [5e-6, 10, 200, 1000]
ns_absolute_max = float('-inf')



def T_SA(i):
    """
    fonction qui calcule la 'temperature liee a l'algorithme pour savoir si on
    va voir ailleurs
    """
    c_r = 1/1200        # cooling rate
    return np.exp(-c_r*i)

def NS(Q, Q_observed, Q_averaged):
    """
    indicateur de proximité du Q en entree avec le Q observe (en donnee de l'exo)
    """
    # ne semble pas avoir de pb 
    
    a = np.sum(  np.power(  np.subtract(Q, Q_observed) , 2)   )
    b = np.sum(  np.power(  np.subtract(Q, Q_averaged) , 2)  )
    
    return 1 - a/b


def opt_param(theta_start = [5e-6, 10, 200, 1000]):
    
    global Q_obs
    global theta_absolute_max
    global ns_absolute_max
    
    
    Q_avg = [np.mean(Q_obs) for i in Q_obs]
    
    theta_old = theta_start.copy()  # initial values of the parameters
    theta_new = theta_old.copy()
    theta_minmax = [[1e-7, 1e-5], 
                    [1, 20],
                    [1, 400],
                    [1, 2000]]      # min/max values of the parameters
    theta_var = [np.diff(i)[0]/20 for i in theta_minmax]   # variance of the parameter, equal to 5% of the range
    
    
    
    # parameters
    ns_old = float("-inf")      # value of the NS coefficient
    n_sim = 0       # nb of simulations yet
    seuil = 0.87    # seuil pour le NS coeff  
    iteration_max = 2e4
    NS_list = np.zeros(int(iteration_max))
    
    
    print("Seuil choisi de : ", seuil)
    print("Starting parameters : ", theta_old)
    
    
    while ns_old < seuil:
        
        # print(n_sim)
        # generate new parameters
        for i in range(4):
            keep_gen = True
            while keep_gen: 
                theta_new[i] = np.random.normal(loc=theta_old[i], scale=theta_var[i])
                
                if theta_new[i] >= theta_minmax[i][0] and theta_new[i] <= theta_minmax[i][1]:
                    keep_gen = False
               
        Q_mod = hydr_model(theta_new[0], theta_new[1], theta_new[2], theta_new[3], precipitation, K_c, 6)[0]
        ns_new = NS(Q_mod, Q_obs, Q_avg)
        NS_list[n_sim] = ns_new
        
        # print(ns_new)
        # print("#"*50)
        # print(ns_old)
        
        if ns_new > ns_absolute_max:
            print("\n    NS maximum absolu amélioré     (iteration " + str(n_sim) + ")")
            print("    NS_max_absolu = ", round(ns_new, ndigits=3))
            print("    Paramètres : ", np.round(theta_absolute_max, decimals=8))
            theta_absolute_max = theta_new.copy()
            ns_absolute_max = ns_new
        
        if ns_new > ns_old:
            print("\nValeur de NS améliorée     (iteration " + str(n_sim) + ")")
            print("NS = ", round(ns_new, ndigits=3))
            theta_old = theta_new.copy()
            ns_old = ns_new
        
        elif np.random.uniform() < np.exp(-abs(ns_new-ns_old)/T_SA(n_sim)):
            print("\nOn va voir ailleurs        (iteration " + str(n_sim) + ")")
            print("NS = ", round(ns_new, ndigits=3))
            
            theta_old = theta_new.copy()
            ns_old = ns_new
        
        if n_sim > iteration_max:
            print("Iterations maximales dépassées pour la boucle principale")
            break
        
        
        n_sim += 1
    
    NS_out = NS_list[:n_sim]
    plt.plot(NS_out, label='NS indicator')
    
    return NS_out, theta_absolute_max




# Valeur empirique trouvée par itération de l'algo
best_param = [9e-7, 4.74, 84.04, 14]








#%% Precipitation modelling




def parametres(P):
    """
    INPUT: 
        - Precipitation [mm/h] (Or other)
    OUTPUT: parameters that describe the rain properties
        -lambda [-]
        -alpha : average precipitation per rainy day [mm - not sure]
        -monthly mean precipitation [mm/h] 
        -monthly standard deviation [mm/h]
    """
    years = int(len(P)/(365*24))
    P_jour = [np.sum(P[24*k:24*k+24]) for k in range(0,years*365) ] #[mm/jour]
    
    n_rainy_day=[0]*12
    I_rain=[0]*12       # [mm/h]
    month_P=[[]]*12

    #I_rain and month_P are the same ??

    for year in range(years):
        for month in range(12):
            for k in range(month_start[month],month_end[month]+1):
                
                x = P_jour[365*year+k]
                month_P[month] = month_P[month]+[x]
                if x != 0:
                    n_rainy_day[month] += 1
                    I_rain[month] += P_jour[365*year+k]
                                    
    lambda_p = [n_rainy_day[k]/(day_month[k]*years) for k in range(12)]
    alpha = [I_rain[k]/n_rainy_day[k] for k in range(12)]
    mean_P = [np.mean(month_P[k]) for k in range(12)]
    std_P = [np.std(month_P[k]) for k in range(12)]
    
    return(lambda_p,alpha,mean_P,std_P)
    
    
############################################



def rain_gen(years=100,plot=True,climate_change=False,alpha_c=alpha_c):
    """
    INPUT:
    - years: number of year we want to simulate
    - plot: do we want to plot the statistics od the simulation ? 
            If yes: plot of lambda, alpha, mean precipitation and mean deviation observed vs generated
    - climate_change : added to compute in a climate change scenario
    """
    #alpha lambda
    #precipitationj=[sum(precipitation[24*k:24*k+24]) for k in range(0,6*365) ]
    lambda_,alpha,mean_P,std_P=parametres(precipitation)
    
    title=" Statistics of the generated Precipitations"
    obs="observed"
    gen="generated"
    
    if climate_change:
        # We are in climate change scenarios, 
        #alpha is modified by the percent of changes in monthly mean rainfall intensity
        alpha_past=alpha.copy()
        lambda_past=lambda_.copy()
        for i in range(0,12):
            alpha[i]=(1+alpha_c[i]/(100))*alpha[i]
            lambda_[i]=(1+lambda_c[i]/100)*lambda_[i]
        title=title+" - climate change scenario"
        obs="current observation"
        gen="future simumation"
            
            
    output = [0 for i in range(365*years)]
    total_day = 0
    
    for y in range(years):
        for m in range(12):
            for d in range(day_month[m]):
                
                # Does it rain ?
                if rd.random() < lambda_[m]:
                    output[total_day] = rd.expovariate(1/alpha[m])
                total_day += 1
                
    P_gen=downscaling(output)
    
    #statistic
    if plot:
        #lambda_gen,alpha_gen,mean_P_gen,std_P_gen=parametres([sum(P_gen[24*k:24*k+24]) for k in range(0,years*365) ])
        lambda_gen,alpha_gen,mean_P_gen,std_P_gen=parametres(P_gen)
        
        figure=plt.figure(figsize=(30,12))
        plt.grid(True)
        plt.subplot(2,2,1)
        plt.subplots_adjust(hspace =0.4,wspace=0.1)
        ax=plt.gca()
        if climate_change:
            plt.plot(month_name,lambda_past,marker='o',label=obs)
        else: 
            plt.plot(month_name,lambda_,marker='o',label=obs)
        plt.plot(lambda_gen,marker='o',label=gen)
        plt.xticks(rotation=50,fontsize=15)
        ax.set_title("Lambda",fontsize=20)
        ax.set_ylabel("[-]")
        ax.legend()
        
        plt.subplot(2,2,2)
        ax=plt.gca()
        plt.subplots_adjust(hspace =0.4,wspace=0.1)
        if climate_change:
            plt.plot(month_name,alpha_past,marker='o',label=obs)
        else: 
            plt.plot(month_name,alpha,marker='o',label=obs)
            
        plt.plot(month_name,alpha_gen,marker='o',label=gen)
        ax.set_title("Alpha",fontsize=20)
        ax.set_ylabel("[mm]")
        plt.xticks(rotation=50,fontsize=15)
        ax.legend()
        
        plt.subplot(2,2,3)
        ax=plt.gca()
        plt.subplots_adjust(hspace =0.4,wspace=0.1)
        plt.plot(month_name,mean_P,marker='o',label=obs)
        plt.plot(month_name,mean_P_gen,marker='o',label=gen)
        ax.set_title("mean precipitation",fontsize=20)
        ax.set_ylabel("average daily precipitation [mm/day]")
        plt.xticks(rotation=50,fontsize=15)
        ax.legend()       
        
        plt.subplot(2,2,4)
        ax=plt.gca()
        plt.subplots_adjust(hspace =0.4,wspace=0.1)
        plt.plot(month_name,std_P,marker='o',label=obs)
        plt.plot(month_name,std_P_gen,marker='o',label=gen)
        ax.set_title(" standard deviation",fontsize=20)
        ax.set_ylabel("[mm/day]")
        plt.xticks(rotation=50,fontsize=15)
        ax.legend()  
        
        plt.suptitle(title,fontsize=30)
        
    return (P_gen)



#%% Dam modelling



def vol_rat_curve(area_rat_curve):
    """
    input : 
        - a list of the area [m2] of the lake for each level [m] of the lake
        area_rat_curve[i] should be equal to the area of the lake at height i meters
    
    Output : 
        - a list of the same size of the input describing the volume [m3] for each
        level. 
        Out[i] is the volume [m3] if the lake is at height i [m]
    """
    
    n = len(area_rat_curve)
    ans = [0 for i in range(n)]
    
    for i in range(1,n):
        ans[i] = (area_rat_curve[i-1] + area_rat_curve[i])/2 + ans[i-1]
    
    return ans

#################################

def lvl_to_vol(level, volume_rating_curve):
    """
    input :
        - a level [m] at which we want to compute the volume. Can be a float.
        - volume_rating_curve a list describing the volume [m3] for each level
        volume_rating_curve[i] should be the volume at lake height i [m]
    
    Output :
        - the volume [m3] at the desired lake height
    """
    
    if level > len(volume_rating_curve):
        raise ValueError("The level is too high ! (" + str(level) + \
                         "m for a maximum level of " + str(len(volume_rating_curve)) + \
                             ")")
    
    # partie entière
    i = int(np.floor(level))
    # partie décimale
    dec = level - i
    
    # bornes de l'interpolation
    a = volume_rating_curve[i]
    b = volume_rating_curve[i+1]
    
    return a + (b-a)*dec
    
###########################################
  
def vol_to_lvl(volume, volume_rating_curve):
    """
    Input :
        - volume [m3] the volume of the reservoir we want to calculate the level
        - volume_rating_curve a list describing the volume [m3] for each level
        volume_rating_curve[i] should be the volume at lake height i [m]
    
    Output :
        - the level [m] corresponding to the given volume and VRC
    """
    
    vrc = volume_rating_curve
    
    if volume > vrc[-1]:
        raise ValueError("The given volume of the reservoir is bigger than \
                         the actual total capacity of the reservoir !" + \
                             str(volume) + " given for a maximum of " + \
                                 str(vrc[-1]))
    
    for i in range(len(vrc)-1):
        if volume >= vrc[i] and volume <= vrc[i+1]:
            dec = (volume - vrc[i]) / (vrc[i+1] - vrc[i])
            return i+dec
    
################################

def Q_S(P,ET):
    """
    Input :
        - P [mm/h] precipitation as a list of length n
        - ET [mm/h] evapotranspiration as a list of length n (same length required !)
    
    Output :
        - the Q_sup [m3/s] as a list of length n required to satisfy the crop
        and city needs (local parameters are given inside the function code)
    """
    ## Returns the discharge that need to supply the city and the crops in m3/h
    
    # Parameters
    A_crop = 5      #[km^2]
    etha_crop=0.8   # [-]
    etha_p = 0.4    # [-]
    
    # variable initialization
    n=len(P)    
    Q_I=[0]*n       # m3/s the needs of the crops
    
    if n != len(ET):
        raise IndexError("Both parameters are not the same length ! (" + \
                         str(n)+" for precipitation vs "+str(len(ET))+ \
                             " for evapotranspiration)")
    
    for i in range(n):
        Q_I[i] = max((( ET[i] - etha_p*P[i])*1e-3/3600*A_crop*(1e6))/etha_crop,0)
        
    Q_city=[Qcity]*n       #[m3/s]
    
    return np.add(Q_I, Q_city)


        
#############################    Reservoir routing


def Q_347(Q, plot=False):
    """
    Input :
        - Q [m3/s] (or another unit) the input discharge
        - plot (default = False) plots the discharge curve and the output Q_347
    
    Output :
        - Q_347 [m3/s] (or the other unit) the discharge that is exceeded 95%
            of the time
    """
    
    n=len(Q)
    sort_Q=sorted(Q,reverse=True)
    
    rank = int(n*95/100)-1
    p_exceedance=[k/n for k in range (1,n+1)]
    Q347=sort_Q[rank]
    if plot:
        #figure=plt.plot(figsize=(15,10))
        plt.semilogy(p_exceedance,sort_Q)
        plt.semilogy(p_exceedance,[Q347]*n,color="red",linestyle='-.',label="Minimum Flow")
        plt.title("Discharge Duration Curve - Minimum Flow = "+str(round(Q347,2)) + " m3/s")
        
        #plt.plot(p_exceedance,[sort_Q[rank]]*(n),color="red")
        plt.xlabel("probability of exceedance [-]")
        plt.ylabel("Discharge [m3/s]")
        plt.legend()
        
    return Q347





#%% reservoir routing




############## MAIN

#parameters of the reservoir
Cqg = 0.6 # [-] sluice gate discharge coefficient
Cqs = 0.7 # [-] spillway discharge coefficient
Lspill = 140 # [m] spillway effective length
p = 19 # [m]  difference between spillway level and minimum level

#parameters of power plant
QT = 30 # [m/s] design discharge of the hydro PP
D = 3.6 # [m]
Lp = 1200 # [m]
ks = 0.5 # [mm]
eta = 0.75 # [-] Careful, more than one eta
Deltaz = 75 # [m] difference in height between the elevation of the empty lake 
                # (headwater) and the tailwater 
lmin_HU = 9 # [m] min height for electricity generation

Qlim = 100 #[m3/s]
g = 9.806 # [m/s2] gravity

f = (1/(-2*np.log(ks/(1000*D*3.7))))**2#Colebrook equation
A_pipe=np.pi*(D/2)**2
kL = f*Lp/(2*g*D) * 2.5 / A_pipe**2     # [m / (m3/s)^2] Loss coefficient


#Power=9806*net_head*Q*eta/1000000    %[MW]
gamma=g*1000
Energy_price=75



# Reservoir routing
def reservoir_routine(Q,P,ET,volume_rating_curve,lmax_HU=15):
    """
    Inputs :
        - Q [m3/s] the flow entering the dam
        - P [mm/h] the precipitation over the whole basin
        - ET [mm/h] the evapotranspiration
        - volume_rating_curve [m] - [m3] the discrete function between the level
            of the lake and its volume
        - lmax_HU [m] maximum level for hydropower production
        
    Output :
        V,l,A_sluice,Q_out,Q_HU,Q_g)
        - V [m3] a list of length n describing the total volume in the dam
        - l [m] a list of length n describing the level in the dam
        - A_sluice [m2] a list of length n describing the open area of the sluice
            to let the ater out in the river
        - Q_out [m3/s] a list of length n describing the flow that exits the
            dam to the river (NOT power generation !)
        - Q_HU [m3/s] a list of length n describing the flow used for power
            generation (does NOT to river)
        - Q_g [m3/s] a list of length n describing the flow that goes through 
            the sluice gate only (NOT the spillway)
        - Pow [Watt] turbine power generation
        - profit [CHF]
        - p_flood
        - E_annual [GWh]
        
    TO-DO :
        - integrate the power generation calculation ? (not sure)
    """
    
    # Variable to calculate
    dt = 3600 # [s] time step integration
    Q347 = Q_347(Q,plot=False)    # [m3/s]
    n = len(Q)
    Q_ind = Q_S(P,ET) # [m3/s]
    
    # variable initialization
    V = [0]*n         # [m3]
    l = [0]*n         # [m] level of the reservoir depending on time
    A_sluice = [0]*n  # [m2] area of the opening of the sluice gate (see Q_g)
    Q_out = [0]*n     # [m3/s] total water out of the dam
    Q_HU = [0]*n      # [m3/s] water going through turbine to generate power
    Q_g = [0]*n       # [m3/s] water flow through the gate
    Pow=[0]*n         # [Watt]
    Y=[0]*n
    
    #Initialization
    l[0] = 14 
    V[0] = lvl_to_vol(l[0], volume_rating_curve)
    Vmax_HU = lvl_to_vol(lmax_HU, volume_rating_curve)
    
    # for 24 hours when is the turbine working (peak hours)
    turbine_hours = [int(x >= 8 and x < 21) for x in range(24)]
    turbine_state = int(l[0] > lmin_HU)     # 1 if the turbine is on for the 
                                            # whole day, 0 otherwise 
    # flood condition an damages
    y_lim = 0.1*np.sqrt(Qlim)+0.5 # maximum y before flood
    Damages=0 # CHF, Damages 
    flood=False # Are we in a flood event or not?
    maxy=0 # maximum of height in a flood event
    n_flood=0
    
    for t in range(n-1):
        
        
        h = t%24
        # is the turbine working during this day ? update if it is midnight
        if h == 0:
            turbine_state = int(l[t] > lmin_HU)
        
        Q_HU[t] = turbine_state * turbine_hours[h] * QT    # [m3/s]
            
        Q_g[t] = max(Q347 , min(Qlim,(V[t]+(Q[t]-Q_HU[t]-Q_ind[t])*dt-Vmax_HU)/dt) )  # [m3/s]
        #print(Cqg*np.sqrt(2*g*l[t]))
        if l[t]==0:
            A_sluice[t]=0
        else:
            A_sluice[t] = Q_g[t] / (Cqg*np.sqrt(2*g*l[t]))  # [m2] 
        
        # does the water spills over the dam ?
        if l[t]<=p:
            Q_out[t] = Q_g[t] #  [m3/s]
        else:
            Q_out[t] = Q_g[t] + Cqs*Lspill*np.sqrt(2*g*(l[t]-p)**3)  #[m3/s]
            
        # update volume and level
        V[t+1]=V[t]+(Q[t]-Q_out[t]-Q_HU[t]-Q_ind[t])*dt
        l[t+1] = vol_to_lvl(V[t+1], volume_rating_curve)
        
        # power generation of the turbine [W]
        HT = l[t] + Deltaz - kL*QT**2   
        Pow[t]=eta*gamma*Q_HU[t]*(HT)
        
        
        # Detection of flood event
        if Q_out[t] > (Qlim + 1):
            n_flood=n_flood+1
            
        y=0.1*np.sqrt(Q_out[t])
        Y[t]=y
            
        if flood: # We are in a flood event
            if y<y_lim:
                # End of the flood event
                z=maxy-y_lim
                maxy=0
                flood=False
                #print(z)
                Damages=Damages+(1+z)**2.25
                #print(Damages)
            else:
                # Still in the flood event
                if y>maxy:
                    maxy=y
                     
        elif y>=y_lim:
            flood=True
            #print("flood")# Beginning of a flood event
            maxy=y

    
    p_flood=n_flood/n
    Total_energy=np.sum(Pow)
    n_year=n/(365*24)
    E_annual=(Total_energy/n_year)*10**(-9)# [GWh]
    profit= Total_energy*Energy_price*10**(-6) - Damages*1000000
    
    return (V,l,A_sluice,Q_out,Q_HU,Q_g,Pow,profit,p_flood,E_annual,Damages)



###Plot
def plot_routine(Q,Q_out,V,l,lmaxHU):
    
    '''
        Input: Qin, Qoutn V, l
        Plot main graph of reservoir routine
    '''
    figure=plt.figure(figsize=(16,10))
    
    plt.subplot(3,1,1)
    ax=plt.gca()
    plt.plot(Q,label="Qin")
    plt.plot(Q_out,label="Qout")
    plt.ylabel("Discharge [m3/s]")
    plt.xlabel("time [hours]")
    ax.legend()
    
    plt.subplot(3,1,2)
    ax=plt.gca()
    plt.plot(V)
    plt.ylabel("Volume [m3]")
    plt.xlabel("time [hours]")
    
    plt.subplot(3,1,3)
    ax=plt.gca()
    plt.plot()
    plt.plot([lmaxHU]*len(Q), label= "max level ",color="red",linestyle="--")
    plt.plot(l)
    
    plt.ylabel("Level [m]")
    plt.xlabel("time [hours]")
    ax.legend()
    
    
    title="Reservoir routine for " +str(int(len(Q)/(365*24)))+" years and lmax = " +str(lmaxHU) + " m"
    
    plt.suptitle(title,fontsize=20)
    
    return None

##Monthly_mean
def monthly_mean(P):
    """
    

    Parameters
    ----------
    P : Table of hourly data.

    Returns: table of mean monthly P
   

    """
    years = int(len(P)/(365*24))
    P_jour = [np.sum(P[24*k:24*k+24]) for k in range(0,years*365) ] #[mm/jour]
    
      # [mm/h]
    month_P=[[]]*12

    #I_rain and month_P are the same ??

    for year in range(years):
        for month in range(12):
            for k in range(month_start[month],month_end[month]+1):
                
                x = P_jour[365*year+k]
                month_P[month] = month_P[month]+[x]

    mean_P = [np.mean(month_P[k]) for k in range(12)]
    return mean_P
    