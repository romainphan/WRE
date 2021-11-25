# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 18:43:12 2021

@author: marie
"""

import Git4


P=rain_gen(years=100,plot=True)
[Q, R, I, s, L, ET]=hydr_model(K_sat, c, t_sub, z, P, K_c,100 , s_0 = 0, V_sup_0 = 0, V_sub_0 = 0)
volume_rating_curve=vol_rat_curve(A_rating)
# Max HU, with max elevation on 15 m
Vmax_HU=lvl_to_vol(15, volume_rating_curve)
print(Vmax_HU)