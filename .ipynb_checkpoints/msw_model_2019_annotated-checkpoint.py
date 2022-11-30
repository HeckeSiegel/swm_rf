#!/usr/bin/env python
# -*- coding: utf-8 -*-

# $Id: sw_1_4_assim.py 520 2015-11-23 09:18:24Z michael.wuersch $

# based upon script from Yvonne and alterated by Kirsten
# from scipy import *
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pdb
from matplotlib import pyplot as plt
import math
from constants import *
from DA_2019 import *
from scipy import stats
from datetime import datetime
from numpy import ndarray
from netCDF4 import Dataset
import numpy.ma as ma
import pickle
from datetime import datetime
from nn_classes import *
import random
from metpy.interpolate import interpolate_to_points

class Rescale:
    '''
    Helper class to pre- and postprocess data
    '''
    def __init__(self,minimum,maximum):
        self.min = minimum
        self.maxmin = maximum - minimum
        
    def to_nn(self,sample_original):
        'Rescale sample from original value to range [0,1] before neural network'
        sample_nn = (sample_original - self.min) / self.maxmin
        return sample_nn
    
    def from_nn(self,sample_nn):
        'Rescale sample after neural network back to original range'
        sample_original = sample_nn*self.maxmin + self.min
        return sample_original

def sweq__(x,nsteps,n,n_ens,mindex,nindex,time,alpha,phic,h_rain):
    """
    x is the concatenated state of size 3*n. This is the relevant model input and ouput.
    nsteps is the number of timesteps
    n is the total grid size
    n_ens is the number of ensemble members
    mindex = mountain index - 1: bell shaped mountain (careful with this)
    nindex = noise index - 1: stochastic Gaussian noise throughout grid domain
    time indicates whether on first time step. Used to install orography and keep count for random seed
    """  
    phi = np.zeros((1,n+2,n_ens))                               # geopotential, returns matrix of 1 row and n+2 columns. Shape (1,n+2)
    beta_new = np.zeros((n+2,n_ens))                            # returns matrix of n+2 columns. Shape (n+2)
    harray = np.zeros((n+2,n_ens))                              # array which will be able to alter the depth of fluid
    y = np.zeros((3*n,n_ens,nsteps))                            # final outputting array
    ku = 2000                                                   # diffusion constant for u
    kh = 6000                                                   # diffusion constant for fluid depth h
    
    if mindex == 1:                                             # bell shaped mountain ridge
      ku = 3000                                                 # diffusion constant for u
      kh = ku                                                   # diffusion coefficient for depth h
      for i in range(n):
        harray[i+1,:] = amp*hw*hw/((float(i)-mu)*(float(i)-mu)+hw*hw)   # every ensemble member has same mountain

    if mindex == 2:                                             # k**(-1) power spectrum
      x_sin = np.arange(0,n)                                    # mindex #1 and #2 need work and checking
      s = 0.2+np.sin((x_sin-100)/500)
      np.random.seed(2)
      data = np.random.rand(n)-0.5                              # data centred around zero
      ps = np.abs(np.fft.fft(data,norm='ortho'))**2             # calculating the power spectrum
      for i in range(n_ens):
        harray[1:n+1,i]=ps*s

    harray[0,:] = harray[n,:]
    harray[n+1,:] = harray[1,:]

    u = np.zeros((3,n+2,n_ens))                                             # horizontal velocity returns matrix of 3 row and n+2 columns 
    h = np.zeros((3,n+2,n_ens))                                             # depth, returns matrix of 3 row and n+2 columns 
    r = np.zeros((3,n+2,n_ens))                                             # rain, returns matrix of 3 row and n+2 columns 
    u[0,1:n+1,:],u[1,1:n+1,:] = x[0:n,:],x[0:n,:]                           # filling in the middle of the u,h and r matrixes with input 
    h[0,1:n+1,:],h[1,1:n+1,:] = x[n:2*n,:],x[n:2*n,:]
    r[0,1:n+1,:],r[1,1:n+1,:] = x[2*n:3*n,:],x[2*n:3*n,:]
    
    u[1,0,:],u[0,0,:] = u[0,n,:],u[0,n,:]                                   # filling in the 2 outer columns of the u, h and r matrixes
    u[1,n+1,:],u[0,n+1,:] = u[0,1,:],u[0,1,:]                               # for boundary conditions from which the model effectively works. 
    h[1,0,:],h[0,0,:] = h[0,n,:],h[0,n,:]
    h[1,n+1,:],h[0,n+1,:] =  h[0,1,:],h[0,1,:]
    r[1,0,:],r[0,0,:] = r[0,n,:],r[0,n,:]
    r[1,n+1,:],r[0,n+1,:] = r[0,1,:],r[0,1,:]

    if time == 0:                                                           # intall ridge - only happens at first time step
      h[:,:,:] = h[:,:,:] - harray[:,:]     

    for it in range(nsteps):                                                # loop over the given model timestep. Saves output after nsteps
      if nindex == 1:
        noise_seed = time+it
        u = noise(u,n,n_ens,n_array,noise_seed)                             # returns the u matrix with the 2nd row with stochastic noise across the ensemble
        u[1,0,:] = u[1,n,:]
        u[1,n+1,:] = u[1,1,:]                                               # BC after u has been added with noise. Otherwise instabilities seen. 

      if mindex == 1 or mindex==2:
        phi[0,1:n+1,:] = np.where( h[1,1:n+1,:]+harray[1:n+1,:] > h_cloud , phic+g*harray[1:n+1,:], g*h[1,1:n+1,:] ) # if condition met, return phic in h matrix. If not, return g*h thing which would be the normal geopotential below the first threshold
      else:
        phi[0,1:n+1,:] = np.where( h[1,1:n+1,:] > h_cloud , phic, g*h[1,1:n+1,:] )

      phi[0,0,:]   = phi[0,n,:]                                             # boundary conditions for geopotential
      phi[0,n+1,:] = phi[0,1,:]
      phi[0,:,:] = phi[0,:,:] + gamma * r[1,:,:] 

      # shallow water equations =D
      u[2,1:n+1,:] = u[0,1:n+1,:] - (dts/(2*dx))*(u[1,2:n+2,:]**2 - u[1,0:n,:]**2) - (2*dts/dx)*(phi[0,1:n+1,:]-phi[0,0:n,:])  + (ku/(dx*dx))*(u[0,2:n+2,:] - 2*u[0,1:n+1,:] + u[0,0:n,:])*dts*2     # momentum equation  # fixed diffusion term
      h[2,1:n+1,:] = h[0,1:n+1,:] - (dts/dx)*(u[1,2:n+2,:]*(h[1,1:n+1,:]+h[1,2:n+2,:]) - u[1,1:n+1,:]*(h[1,0:n,:]+h[1,1:n+1,:])) + (kh/(dx*dx))*(h[0,2:n+2,:] - 2*h[0,1:n+1,:] + h[0,0:n,:])*dts*2   # continuity equation  # fixed diffusion term
      
      if mindex == 1 or mindex==2:
        mask = np.logical_and(h[1,1:n+1,:]+harray[1:n+1,:] > h_rain, u[1,2:n+2,:]-u[1,1:n+1,:] < 0)    # conditions for rain
      else:
        mask = np.logical_and(h[1,1:n+1,:] > h_rain, u[1,2:n+2,:]-u[1,1:n+1,:] < 0)
      beta_new[1:n+1,:] = np.where( mask, beta , 0 )
    
      for i in range(n_ens):  
          r[2,1:n+1,i] = r[0,1:n+1,i] - (dts/(2*dx))*(u[1,2:n+2,i]+u[1,1:n+1,i])*(r[1,2:n+2,i]-r[1,0:n,i]) - alpha[i]*dts*2.0*r[1,1:n+1,i]-2.0*beta_new[1:n+1,i]*(dts/dx)*(u[1,2:n+2,i]-u[1,1:n+1,i]) + (kr/(dx*dx))*(r[0,2:n+2,i] - 2.0*r[0,1:n+1,i] + r[0,0:n,i])*dts*2 # rain equation  # with advection
      #r[2,1:n+1,:] = r[0,1:n+1,:] - alpha*dts*2.0*r[1,1:n+1,:]-2.0*beta_new[1:n+1,:]*(dts/dx)*(u[1,2:n+2,:]-u[1,1:n+1,:]) + (kr/(dx*dx))*(r[0,2:n+2,:] - 2.0*r[0,1:n+1,:] + r[0,0:n,:])*dts*2 # rain equation, no advection
      #r[2,1:n+1,:] = r[0,1:n+1,:] - alpha*dts*2.0*r[1,1:n+1,:]-2.0*beta_new[1:n+1,:]*(dts/dx)*(u[1,2:n+2,:]-u[1,1:n+1,:]) + (kr/(dx*dx))*(r[0,2:n+2,:] - 2.0*r[0,1:n+1,:] + r[0,0:n,:])*dts*2  - (dts/(dx))*(u[1,1:n+1,:])*(r[1,2:n+2,:]-r[1,0:n,:])  # other advection term 
     
      u[2,0,:]   = u[2,n,:]                                                  # boundary conditions
      u[2,n+1,:] = u[2,1,:]
      h[2,0,:]   = h[2,n,:]
      h[2,n+1,:] = h[2,1,:]
      r[2,0,:]   = r[2,n,:]
      r[2,n+1,:] = r[2,1,:]
      r[2,:,:] = np.where(r[2,:,:]<0.,0,r[2,:,:])

      d = filter_parameter*.5*(u[2,:,:] - 2.*u[1,:,:] + u[0,:,:])            # RAW filter. Accounts for the growing computational mode.
      u[0,:,:] = u[1,:,:] + alpha_filt*d
      u[1,:,:] = u[2,:,:] - (1-alpha_filt)*d                     
      d = filter_parameter*.5*(h[2,:,:] - 2.*h[1,:,:] + h[0,:,:])
      h[0,:,:] = h[1,:,:] + alpha_filt*d
      h[1,:,:] = h[2,:,:] - (1-alpha_filt)*d
      d = filter_parameter*.5*(r[2,:,:] - 2.*r[1,:,:] + r[0,:,:])
      r[0,:,:] = r[1,:,:] + alpha_filt*d
      r[1,:,:] = r[2,:,:] - (1-alpha_filt)*d

      y[0:n,:,it] = u[2,1:n+1,:]
      y[n:2*n,:,it] = h[2,1:n+1,:]
      y[2*n:3*n,:,it] = r[2,1:n+1,:] 

    return y

def Initialize__(n,n_ens,mindex,nindex,alpha,phic,h_rain):
  '''
  Initialise x array input for sweq function when initialising DA. 
  n - grid size 
  nens - number of ensembles. Note!: no multiplying factor here since 'Initialize' used for DA only.
  '''
  state = np.zeros((3*n,n_ens))
  state[0:n] = np.zeros((n,n_ens))+0.                          # velocity 0m/s
  state[n:2*n] = np.zeros((n,n_ens))+90.                       # fluid depth 90m

  state = sweq__(state,1000,n,n_ens,mindex,nindex,time=0,alpha=alpha,phic=phic,h_rain=h_rain)      # state to begin with. 1000 nsteps taken
  return state

def noise(u,n,n_ens,n_array,noise_seed):   
    ''' 
    Random stochastic noise added to the velocity. Mimics turbulence in the boundary layer. 
    n_array tells model where noise is located. narray.shape = (number of perturbation areas, features of pert area)
    features of pert area = [half width, amplitude, start, end of noise field region]
    iteration added so 'random' noise is seeded but different on each iteration cycle
    '''
    unoise = np.zeros((2*n,n_ens))
    mu = float((n+1.0)/2.0)                                           # center of noise

    for area in range(len(n_array[:,0])):                             # loop goes over all areas of perturbations 
      sig = n_array[area,0]                                           # sigma, half width of noise field
      amp = n_array[area,1]                                           # amplitude of noise field
      d = np.array(range(n+1))
      z = (1.0/(sig*np.sqrt(2.0*np.pi)))*np.exp(-0.5*((d-mu)/sig)**2)
      zsum = z[1:n+1]-z[0:n]                                          # zsum will look like convergence
      zsum = amp*zsum/max(zsum)                                       # normalising

      for e in range(n_ens):                                          # adds this noise to all ensembles
        start = n_array[area,2]
        end = n_array[area,3]
        noise_seed_chosen = ((noise_seed+e)*(e+2))-(noise_seed+e*3)   # choosing the random seed so that each ensemble member is perturbed differently, and at different times differently also. But same seed chosen for all orographic features.
        np.random.seed(noise_seed_chosen)  
        pos = np.random.randint(start,end)                            # each ensemble will have own random noise added within bounds created
        unoise[pos:pos+n,e] = unoise[pos:pos+n,e] + zsum
        unoise[0:n,e] = unoise[0:n,e] + unoise[n:n+n,e]
        unoise[n:n+n,e] = 0 
        u[1,1:n+1,e] = u[1,1:n+1,e] + unoise[0:n,e]
    return u

def interp_bound(x_interp):
    '''
    Smooth nan bounds
    '''
    
    x_interp[np.where(np.isnan(x_interp))] = np.nanmean(x_interp).reshape(-1)
    
    return x_interp

def interp_obs(obs,H,radar):
    '''
    Interpolate initial observations for not fully observed grid
    '''
    # split observed grid into u,h,r
    u = 0
    while(H[u]<250):
        u += 1
    obs_points_u = H[:u]
    h = u
    while(H[h]<500):
        h += 1
    obs_points_h = H[u:h]
    obs_points_r = H[h:]
    
    # grids to be interpolated
    points_u = np.arange(0,250)
    points_h = np.arange(250,500)
    
    # observations
    obs_sparse_u = obs[obs_points_u]
    obs_sparse_h = obs[obs_points_h]
    
    # interpolation
    res_u = interpolate_to_points(obs_points_u,obs_sparse_u,points_u,interp_type='linear')
    res_h = interpolate_to_points(obs_points_h,obs_sparse_h,points_h,interp_type='linear')
    
    # smooth nan boundarys
    res_u = interp_bound(res_u)
    res_h = interp_bound(res_h)
    
    if(radar==False):
        points_r = np.arange(500,750)
        obs_sparse_r = obs[obs_points_r]
        res_r = interpolate_to_points(obs_points_r,obs_sparse_r,points_r,interp_type='linear')
        res_r = interp_bound(res_r)
    else:
        res_r = np.zeros(250)
        res_r[obs_points_r-500] = obs[obs_points_r]
        
    res   = np.zeros(3*n)
    res[:250], res[250:500], res[500:] = res_u, res_h, res_r
    return res

def get_radar_obs(obs,wind_frac):
    '''
    Determine which grid points have value r>0.005 and are thus observed and add additional wind_frac % of wind observations
    '''
    obs_wet = np.argwhere(obs[500:] > 5e-5).reshape(-1)
    obs_dry = np.argwhere(obs[500:] <= 5e-5).reshape(-1)
    num_dry = int(len(obs_dry)*wind_frac)
    obs_wind = np.array(sorted(random.sample(obs_dry.tolist(), num_dry)))
    H = np.array(np.sort(np.concatenate([obs_wet,obs_wind,obs_wet+250,obs_wet+500,]))).astype(int)
    rdiag = np.zeros((3*n),dtype='float32')                     
    rdiag_inv = np.zeros((3*n),dtype='float32')
    for i in range(3):
        rdiag[i*n:(i+1)*n] = sig[i]**2
        rdiag_inv[i*n:(i+1)*n] = 1./sig[i]**2
    rdiag = rdiag[H]
    rdiag_inv = rdiag_inv[H]
    
    return H, rdiag, rdiag_inv
