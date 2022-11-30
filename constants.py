#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:45:23 2019

@author: Yvonne.Ruckstuhl, edited by Kirsten ;) 
"""
import numpy as np

n = 250                                                     # number of gridpoints
dte = 60                                                    # number of timesteps in a DA cycle  
dte_train = 60
multiply = 1                                                # factor by which to increase ensemble size after DA initialisation
save_directory = 'data/DA/'                                 # directory for saving output
dx = 500.0                                                  # horizontal resolution of 500
dy = 500.0                                                  # vertical resolution
dts = 4.0                                                   # timestep used in discretisation
g = 10                                                      # gravitational constant
h_cloud = 90.02                                             # height threshold for cloud formation
h_0 = 90.0                                                  # resting height of fluid
gamma = g*h_0                                               # weight for negative buoyancy of rain (c in paper)
beta = 0.1                                                  # lag between cloud and rain formation. standard 1/300
#alpha = 0.00014                                             # half-life of influence of rain of roughly 1 hour was 0.00067
kr = 10.0                                                   # diffusion constant for rain.
hw = 5                                                      # half width of mountain ridge
amp = 1.2                                                   # amplitude of mountain ridge
mu = n/2                                                    # centre point of mountain ridge
t = 17280                                                   # length of the model simulation which uses discretisation timestep dts
filter_parameter = 0.7                                      # For RAW filter to reduce computational mode.
alpha_filt = 0.53                                           # usually 0.53. Want just above 0.5 to be conditionally stable. For RAW filter.


sig = [0.001, 0.01, 0.0001]                                 # [0.001,  0.01, 0.0001]
loc_radius = 3                                              # normally 3
mindex = 0                                                  # mountain index - 0: flat orography, 1: bell-shaped mountain, 2: power spectrum with sine envelope
nindex = 1                                                  # noise index - 0: no random pertubational noise in boundary layer, 1: random noise
n_array = np.array([[4,0.009,0,n]])                         # info of random noise if nindex = 1 Each row indicates a location of noise in the order 
                                                            # [sigma (half width) of the noise field,amplitude, start, end of place for model to choose 
                                                            # of noise field]. Currently perturbations in whole domain like original (first row). 
                                                            # Standard amplitude = 0.005. change to 0.000005 if mindex = 1.

nparams = 3
dec = [5,5,4] 
bounds_param = [[0.0003,0.001],[899.7,899.9],[90.15,90.25]] # bounds for parameters
bounds_uhr = [[-0.2,0.2],[89.5, 91.25],[0.0,0.002]]         # bounds for u,h,r
granularity = [3,3,3]                                       # number of values for alpha, phi, h within bounds
start_t = 0                                                 # start time for test data