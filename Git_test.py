# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 18:43:12 2021

@author: marie
"""

from Git4 import * 

P=rain_gen(years=100,plot=True)
volume_rating_curve=vol_rat_curve(A_rating)
[Q, R, I, s, L, ET]=hydr_model(K_sat, c, t_sub, z, P, K_c, n_years=100, s_0 = 0, V_sup_0 = 0, V_sub_0 = 0)

def reservoir_(Q,P,ET,volume_rating_curve):
    """
    Inputs :
        - Q [m3/s] the flow entering the dam
        - P [mm/h] the precipitation over the whole basin
        - ET [mm/h] the evapotranspiration
        - volume_rating_curve [m] - [m3] the discrete function between the level
            of the lake and its volume
        
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
    
    #Initialization
    l[0] = 19
    V[0] = lvl_to_vol(V[0], volume_rating_curve)
    Vmax_HU = lvl_to_vol(15, volume_rating_curve)
    
    # for 24 hours when is the turbine working (peak hours)
    turbine_hours = [int(x >= 8 and x < 21) for x in range(24)]
    turbine_state = int(l[0] > lmin_HU)     # 1 if the turbine is on for the 
                                            # whole day, 0 otherwise 
    
    for t in range(n):
        
       
        
        h = t%24
        # is the turbine working during this day ? update if it is midnight
        if h == 0:
            turbine_state = int(l[t] > lmin_HU)
        
        Q_HU[t] = turbine_state * turbine_hours[h] * QT    # [m3/s]
            
        Q_g[t] = max(Q347 , min(Qlim, \
                    (V[t]+(Q[t]-Q_HU[t]-Q_ind[t])*dt-Vmax_HU)/dt) )  # [m3/s]
        
        print(t,l[t])
        A_sluice[t] = Q_g[t] / (Cqg*np.sqrt(2*g*l[t]))  # [m2] 
        
        # does the water spills over the dam ?
        if l[t]<=p:
            Q_out[t] = Q_g[t] #  [m3/s]
        else:
            Q_out[t] = Q_g[t] + Cqs*Lspill*np.sqrt(2*g*(l[t]-p)**3)  #[m3/s]
            
        # update volume and level
        V[t+1]=V[t]+(Q[t]-Q_out[t]-Q_HU[t]-Q_ind[t])*dt
        print(V[t+1])
        l[t+1] = vol_to_lvl(V[t+1], volume_rating_curve)
        print(l[t+1])
        
    return (V,l,A_sluice,Q_out,Q_HU,Q_g)


[V,l,A_sluice,Q_out,Q_HU,Q_g]=reservoir_(Q,P,ET,volume_rating_curve)