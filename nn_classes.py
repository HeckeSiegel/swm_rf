#!/usr/bin/env python
# -*- coding: utf-8 -*-

# $Id: sw_1_4_assim.py 520 2015-11-23 09:18:24Z michael.wuersch $

# based upon script from Yvonne and alterated by Kirsten
import numpy as np
from constants import *
from msw_model_2019_annotated import *
from DA_2019 import *
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from netCDF4 import Dataset as Dataset_net
from metpy.interpolate import interpolate_to_points
from sklearn.linear_model import LinearRegression
import os
import seaborn as sn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from tqdm.auto import trange, tqdm
from pyro import poutine
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import forestci as fci

def create_folder_path(path):
    MYDIR = (path)
    CHECK_FOLDER = os.path.isdir(MYDIR)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)

def save_hparam(obj, version):
    with open('models/'+version+'_hparam.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_hparam(version):
    with open('models/'+version+'_hparam.pkl', 'rb') as f:
        return pickle.load(f)

class DataGenerator_uniform:
    '''
    First, generates initial truth state + initial analysis states (with <nens> ensemble members) with <cycles> data assimilation cycles and saves it in 
    'data/init_truth.csv', 'data/init_analysis.csv'. From these states, generates data for the neural network training with n_train training samples and n_val  
    validation samples taken from uniform distributions with bounds [0,1] and saves it as csv files in 'data/background/<id>.csv' (training) and 
    'data/analysis/<id>.csv' (validation) with (u1,..,u250,h1,..,h250,..,r1..,r250) as columns. Truth states are saved in 'data/truth/<id>.csv' to use for further
    data assimilation experiments.
    
    nens: data assimilation ensemble members to generate initial analysis states
    cycles: data assimilation cycles to initialise analysis and truth states
    n_train: number of training samples (code only works if: n_train >= nens and n_train mod nens = 0)
    n_val: number of validation/test samples (no resctrictions)
    mode: either 'val' or 'test'
    '''
    def __init__(self,nens,cycles,n_train,n_val,mode):
        
        # initialize random truth + background states with shallow water model for 1000 time steps
        parameters_truth = np.array([bounds_param[i][0] for i in range(3)])
        parameters_bg = np.zeros((nens,3))
        for i in range(len(bounds_param)):
            parameters_bg[:,i] = np.around(np.random.uniform(bounds_param[i][0],bounds_param[i][1],nens),5)
        truth = Initialize__(n,1,mindex,nindex,[parameters_truth[0]],[parameters_truth[1]],[parameters_truth[2]])[:,:,999]
        background = Initialize__(n,nens,mindex,nindex,parameters_bg[:,0],parameters_bg[:,1],parameters_bg[:,2])[:,:,999]
        analysis = 1.0*background
        
        # save random initial truth and background for plotting
        create_folder_path('data')
        
        # go through <cycles> data assimilation cycles using <nens> ensemble members
        for i in range(cycles):
            truth = sweq__(truth,dte,n,1,mindex,nindex,1000+(i+1)*dte,[parameters_truth[0]],[parameters_truth[1]],[parameters_truth[2]])[:,:,-1]
            background = sweq__(analysis,dte,n,nens,mindex,nindex,1000+(i+1)*dte,parameters_bg[:,0],parameters_bg[:,1],parameters_bg[:,2])[:,:,-1]

            truth_obs = 1.0*truth.reshape(-1)
            obs_ncyc = i 
            obs = get_obs(truth=truth_obs,seed=obs_ncyc)
            H, rdiag, rdiag_inv = get_radar_obs(obs,0.25)

            analysis = EnKF(obs,background,obs_ncyc,nens,H,rdiag)
        obs_ncyc += 1
        # save truth and analysis to start offline training generation from
        np.savetxt('data/H.csv',H,delimiter=',',fmt='%d')
        np.savetxt('data/init_truth_'+str(nens)+'.csv',truth.reshape((1,750)),delimiter=',',fmt=['%f']*750)
        np.savetxt('data/init_analysis_'+str(nens)+'.csv',analysis,delimiter=',',fmt=['%f']*nens)
        print(len(H)/750)
        # read in labels
        labels_train = (np.loadtxt('data/train_labels_raw.csv',delimiter=',',skiprows=1)[:n_train,:]).T
        labels_val = (np.loadtxt('data/'+mode+'_labels_raw.csv',delimiter=',',skiprows=1)[:n_val,:]).T
        
        # if n_train > nens initial background state has to be inflated first for shallow water model
        analysis_old = np.loadtxt('data/init_analysis_'+str(nens)+'.csv',delimiter=',')
        if nens<n_train:
            analysis_old = np.repeat(analysis_old,n_train/nens,1)
        
        # generates training data and saves it in 'data/background/<id>.csv'
        create_folder_path('data/background')
        data_train = sweq__(1.0*analysis_old,dte_train,n,n_train,mindex,nindex,1000+obs_ncyc*dte,labels_train[0,:],labels_train[1,:],labels_train[2,:])[:,:,-1]
        train_sample = np.zeros((1,750))
        train_total = np.zeros((labels_train.shape[1],750))
        for i in range(labels_train.shape[1]):
            train_sample[0,:] = data_train[:,i]
            train_total[i,:] = data_train[:,i]
            np.savetxt('data/background/'+str(i)+'.csv',train_sample,delimiter=',',fmt=['%f']*3*n)
        np.savetxt('data/background/train.csv',train_total,delimiter=',',fmt=['%f']*3*n)
        
        # true atmosphere states propagate forward in time
        truth = np.loadtxt('data/init_truth_'+str(nens)+'.csv',delimiter=',')
        truth_expanded = np.repeat(np.expand_dims(truth,axis=0),n_val,0).T
        truth_prop = sweq__(1.0*truth_expanded,dte_train,n,n_val,mindex,nindex,1000+obs_ncyc*dte,labels_val[0,:],labels_val[1,:],labels_val[2,:])[:,:,-1]
        
        # generate validation/test data and saves it in 'data/analysis/<id>.csv'
        spinup_an = np.loadtxt('data/init_analysis_'+str(nens)+'.csv',delimiter=',')
        data_val = sweq__(1.0*spinup_an,dte_train,n,nens,mindex,nindex,1000+obs_ncyc*dte,labels_train[0,:nens],labels_train[1,:nens],labels_train[2,:nens])[:,:,-1]
        
        val_sample = np.zeros((nens,750))
        truth_sample = np.zeros((1,750))
        obs_samples = np.zeros((nens,750))
        create_folder_path('data/truth')
        create_folder_path('data/analysis')
        for i in range(labels_val.shape[1]):
            truth_sample[0,:] = truth_prop[:,i]
            np.savetxt('data/truth/'+str(i)+'.csv',truth_sample,delimiter=',',fmt=['%f']*3*n)
            H, rdiag, rdiag_inv = get_radar_obs(truth_prop[:,i].reshape(-1),0.25)
            obs = get_obs(truth_prop[:,i].reshape(-1),i)
            analysis = EnKF(obs,data_val,i,nens,H,rdiag)
            val_sample[:,:] = analysis.T
            np.savetxt('data/analysis/'+str(i)+'.csv',val_sample,delimiter=',',fmt=['%f']*3*n)
        # saves labels for training and validation
        np.savetxt('data/background/labels_raw.csv',labels_train.T,delimiter=',',fmt=['%f','%f','%f'],header='alpha,phi,h',comments='')
        np.savetxt('data/truth/labels_raw.csv',labels_val.T,delimiter=',',fmt=['%f','%f','%f'],header='alpha,phi,h',comments='')
        
        # rescales labels to [0,1] for neural network training
        Preprocessor('background')
        Preprocessor('truth')
                
class DataGenerator_ml_init:
    '''
    Initializes data (parameters+atmospheric states) for further data assimilation cycles using trained neural network.
    
    nens: number of data assimilation/background ensemble members
    samples: number of samples the bnn predicts for each of the nens inputs
    exp_ID: experiment ID, refers to ground truth parameters (will be constant in time for each experiment)
    '''
    def __init__(self,ml_model,nens,samples,exp_ID,n_train):
        # offline trained neural network (saved in 'models/0_model.pt','models/0_params.pt') is used to predict model parameters from inerpolated observations
        create_folder_path('models')
        version = str(nens)
        if ml_model=='bnn':
            hparams = load_hparam(version)
            model = BNN_fc(hparams)
            saved_model_dict = torch.load('models/'+version+'_model.pt')
            saved_guide_dict = torch.load('models/'+version+'_params.pt')

            model.load_state_dict(saved_model_dict['model'])
            guide = saved_model_dict['guide']
            pyro.get_param_store().load('models/'+version+'_params.pt')
            model.eval()

            predictive = Predictive(model, guide=guide, num_samples=samples)
            x = torch.from_numpy(np.loadtxt('data/analysis/'+str(exp_ID)+'.csv',delimiter=','))
            y = (predictive(x.float())['obs'].detach().numpy().squeeze()).reshape((-1,3))
            
        else:
            rf_model = pickle.load(open("models/"+version+"rf", 'rb')) # load random forest model
            x = np.loadtxt('data/analysis/'+str(exp_ID)+'.csv',delimiter=',')
            y = np.array([tree.predict(x) for tree in rf_model])
            y = y.reshape((nens*samples,3))
        
        y_sample = y[np.random.randint(nens*samples, size=nens),:]
        y_sample = (y_sample - np.mean(y_sample,axis=0).reshape((-1,3))) * (np.std(y,axis=0).reshape((-1,3))/np.std(y_sample,axis=0).reshape((-1,3))) + np.mean(y,axis=0).reshape((-1,3))
            
        
        # create folders if not there already
        create_folder_path('data/online')
        create_folder_path('data/online/analysis')
        create_folder_path('data/online/background')
        create_folder_path('data/online/metrics')
        create_folder_path('data/online/truth')
        # parameter predictions are saved in 'data/online/background/labels_rescaled.csv' to use for next background state
        np.savetxt('data/online/background/labels_rescaled.csv',y,delimiter=',',fmt=['%f','%f','%f'],header='alpha,phi,h',comments='')
        
        # truth parameters are constant in time
        labels_truth_raw = np.loadtxt('data/truth/labels_raw.csv',delimiter=',',skiprows=1)[exp_ID,:]
        np.savetxt('data/online/truth/labels_raw.csv',labels_truth_raw.reshape((1,3)),delimiter=',',fmt=['%f','%f','%f'],header='alpha,phi,h',comments='')
        
        # rescale parameters from [0,1] to actual interval to use in shallow water model
        post = Postprocessor()
        y_re = np.zeros((nens,3))
        y_re_sample = np.zeros((nens,3))
        for i in range(nens):
            y_re[i,:] = post.rescale(y[i,:])
            y_re_sample[i,:] = post.rescale(y_sample[i,:])
        np.savetxt('data/online/background/labels_raw.csv',y_re,delimiter=',',fmt=['%f','%f','%f'],header='alpha,phi,h',comments='')
        
        # Initialize truth and analysis with ground truth/predicted parameters
        truth = Initialize__(n,1,mindex,nindex,[labels_truth_raw[0]],[labels_truth_raw[1]],[labels_truth_raw[2]])[:,:,999]
        background = Initialize__(n,nens,mindex,nindex,y_re_sample[:,0],y_re_sample[:,1],y_re_sample[:,2])[:,:,999]
        analysis = 1.0*background
        
        # one DA cycle
        truth = sweq__(truth,dte,n,1,mindex,nindex,1000+dte,[labels_truth_raw[0]],[labels_truth_raw[1]],[labels_truth_raw[2]])[:,:,-1]
        np.savetxt('data/online/truth/0.csv',truth.reshape((1,-1)),delimiter=',',fmt=['%f']*3*n)
        background = sweq__(analysis,dte,n,nens,mindex,nindex,1000+dte,y_re_sample[:,0],y_re_sample[:,1],y_re_sample[:,2])[:,:,-1]

        truth_obs = 1.0*truth.reshape(-1)
        obs = get_obs(truth=truth_obs,seed=0)
        H, rdiag, rdiag_inv = get_radar_obs(obs,0.25)
        analysis = EnKF(obs,background,0,nens,H,rdiag)
        
        for i in range(nens):
            np.savetxt('data/online/analysis/'+str(i)+'.csv',analysis[:,i].reshape((1,-1)),delimiter=',',fmt=['%f']*3*n)
            
class DataGenerator_const_init:
    '''
    Initializes parameters and atmospheric states for data assimilation experiments withouth parameter prediction.
    
    nens: number of data assimilation/background ensemble members
    exp_ID: experiment ID, refers to ground truth parameters (will be constant in time for each experiment)
    mode: either 'true' (= truth parameters are known and used for background), 'false' (= truth parameters are not known and not estimated)
    '''
    def __init__(self,nens,exp_ID,mode):
        
        # truth parameters are constant in time
        labels_truth_raw = np.loadtxt('data/truth/labels_raw.csv',delimiter=',',skiprows=1)[exp_ID,:]
        np.savetxt('data/online/truth/labels_raw.csv',labels_truth_raw.reshape((1,3)),delimiter=',',fmt=['%f','%f','%f'],header='alpha,phi,h',comments='')
        
        y = np.ones((nens,3))
        if mode=='true':
            # truth parameters are known
            y *= labels_truth_raw.reshape((1,3))
        elif mode=='false':
            # random parameters from training set are used
            y_random = np.loadtxt('data/train_labels_raw.csv',delimiter=',',skiprows=1)[exp_ID,:].reshape((1,3))
            y *= y_random
        np.savetxt('data/online/background/labels_raw.csv',y,delimiter=',',fmt=['%f','%f','%f'],header='alpha,phi,h',comments='')
        
        # Initialize truth and analysis with ground truth/predicted parameters
        #truth = np.loadtxt('data/truth/'+str(exp_ID)+'.csv',delimiter=',').reshape((-1,1))
        #analysis = np.loadtxt('data/analysis/'+str(exp_ID)+'.csv',delimiter=',').T
        
        truth = Initialize__(n,1,mindex,nindex,[labels_truth_raw[0]],[labels_truth_raw[1]],[labels_truth_raw[2]])[:,:,999]
        background = Initialize__(n,nens,mindex,nindex,y[:,0],y[:,1],y[:,2])[:,:,999]
        analysis = 1.0*background
        
        # one DA cycle
        truth = sweq__(truth,dte,n,1,mindex,nindex,1000+dte,[labels_truth_raw[0]],[labels_truth_raw[1]],[labels_truth_raw[2]])[:,:,-1]
        np.savetxt('data/online/truth/0.csv',truth.reshape((1,-1)),delimiter=',',fmt=['%f']*3*n)
        background = sweq__(analysis,dte,n,nens,mindex,nindex,1000+dte,y[:,0],y[:,1],y[:,2])[:,:,-1]

        truth_obs = 1.0*truth.reshape(-1)
        obs = get_obs(truth=truth_obs,seed=0)
        H, rdiag, rdiag_inv = get_radar_obs(obs,0.25)
        analysis = EnKF(obs,background,1,nens,H,rdiag)
        
        for i in range(nens):
            np.savetxt('data/online/analysis/'+str(i)+'.csv',analysis[:,i].reshape((1,-1)),delimiter=',',fmt=['%f']*3*n)
    
class DataGenerator_const:
    '''
    Generates data for data assimilation experiments with constant parameters.
    
    nens: number of data assimilation/background ensemble members
    DA_cycle: current data assimilation cycle
    mode: either 'true' (= truth parameters are known and used for background) or 'false' (= truth parameters are not known and not estimated) or 
    'bnn' (= parameters were estimated with bnn)
    '''
    def __init__(self,nens,DA_cycle,mode):
        
        # use old analysis to generate new background from
        if mode in ['bnn','rf']:
            labels_background_full = np.loadtxt('data/online/background/labels_raw.csv',delimiter=',',skiprows=1)
            index = np.random.choice(labels_background_full.shape[0], nens, replace=False)
            labels_background = labels_background_full[index,:]
            labels_background = (labels_background - np.mean(labels_background,axis=0).reshape((-1,3))) * (np.std(labels_background_full,axis=0).reshape((-1,3))/np.std(labels_background,axis=0).reshape((-1,3))) + np.mean(labels_background_full,axis=0).reshape((-1,3))
        else:
            labels_background = np.loadtxt('data/online/background/labels_raw.csv',delimiter=',',skiprows=1)[:nens,:]
        analysis_old = np.zeros((750,nens))
        for i in range(nens):
            analysis_old[:,i] = np.loadtxt('data/online/analysis/'+str(i)+'.csv',delimiter=',')
        background = sweq__(analysis_old,dte,n,nens,mindex,nindex,1000+DA_cycle*dte,labels_background[:,0],labels_background[:,1],labels_background[:,2])[:,:,-1]
        # propagate truth in time for dte time steps
        truth_old = np.loadtxt('data/online/truth/0.csv',delimiter=',').reshape((-1,1))
        labels_truth = np.loadtxt('data/online/truth/labels_raw.csv',delimiter=',',skiprows=1)
        truth_new = sweq__(truth_old,dte,n,1,mindex,nindex,1000+DA_cycle*dte,[labels_truth[0]],[labels_truth[1]],[labels_truth[2]])[:,:,-1]
        np.savetxt('data/online/truth/0.csv',truth_new.reshape((1,-1)),delimiter=',',fmt=['%f']*3*n)
        
        # calculate analysis and observations for new parameter predictions
        obs = get_obs(truth=truth_new.reshape(-1),seed=0)
        H, rdiag, rdiag_inv = get_radar_obs(obs,0.25)
        analysis = EnKF(obs,background,0,nens,H,rdiag)
        for i in range(nens):
            np.savetxt('data/online/analysis/'+str(i)+'.csv',analysis[:,i].reshape((1,-1)),delimiter=',',fmt=['%f']*3*n)

def train_rf(nens,n_estimator,max_depth,min_samples_split,ccp_alpha,val_obs,random_seed):
    hparams = {
        'version' : str(nens),
        'n_estimators' : n_estimator,
        'max_depth' : max_depth,
        'random_seed': random_seed,
        'min_samples_split' : min_samples_split,
        'ccp_alpha' : ccp_alpha
    }
    
    if val_obs:
        H = np.loadtxt('data/H.csv',delimiter=',',dtype='int')
    else:
        H = np.arange(750)
    
    # load training data
    x_train = np.loadtxt("data/background/train.csv", delimiter=',')[:,H]
    y_train = np.loadtxt("data/background/labels_rescaled.csv", delimiter=',', skiprows=1)
    # train the model
    model_rf = RandomForestRegressor(min_samples_split=hparams["min_samples_split"], max_depth=hparams["max_depth"], n_estimators=hparams["n_estimators"], random_state=hparams["random_seed"],ccp_alpha=hparams["ccp_alpha"],oob_score=True)
    model_rf.fit(x_train, y_train)

    # load test
    x_test = np.zeros((100,len(H)))
    for i in range(100):
        x_test[i,:] = np.loadtxt("data/truth/"+str(i)+".csv", delimiter=',')[H]
        
    y_test = np.loadtxt("data/val_labels_rescaled.csv", delimiter=',', skiprows=1)[:100,:]
    
    print("oob score", model_rf.oob_score_)
    print("test score", model_rf.score(x_test,y_test))
    
    pickle.dump(model_rf, open("models/"+str(nens)+"rf", 'wb'))


def train_bnn(nens,epochs,val_obs,neurons):
    '''
    Trains bnn on data generated with nens data assimilation ensemble members.
    nens: number of analysis/background ensemble members used to generate training data
    epochs: number of epochs for bnn training
    samples: number of samples bnn draws for each input
    '''
    pyro.clear_param_store()
    hparams = {
        'version' : str(nens),
        'hidden_layers' : [neurons,neurons,neurons],
        'lr' : 1e-3,
        'batch_size' : 32,
        'batch_size_val' : 1,
        'num_workers' : 6,
        'sigma' : 0.001,
        'fraction train' : 1.0,
        'fraction val' : 1.0,
        'online' : '',
        'train cycle' : 0,
        'obs' : val_obs
    }

    train_loader, val_loader = DataModule(hparams).get_data()

    model = BNN_fc(hparams)
    guide = AutoDiagonalNormal(model)
    adam = pyro.optim.Adam({"lr": hparams['lr']})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    log_dir = '/project/meteo/work/S.Legler/swm_rf/logs/'
    model_dir = '/project/meteo/work/S.Legler/swm_rf/models/'
    create_folder_path('logs')
    
    # train bnn
    train_loss_log = np.zeros(epochs)
    val_loss_log = np.zeros((epochs,3*2))

    for j in range(epochs):
        loss = 0

        model.train()
        for batch_id, data in enumerate(train_loader):
            # calculate the loss and take a gradient step
            #print(batch_id,datetime.now().strftime("%H:%M:%S"))
            loss += svi.step(data[0].float(), data[1].float())

        if j%10 == 0:
            normalizer_train = len(train_loader.dataset)
            total_epoch_loss_train = loss / normalizer_train

            train_loss_log[j] = total_epoch_loss_train
            model.eval()
            val_loss_epoch, val_spread_epoch, _, _ = val_loss(model,guide,val_loader,hparams,5,nens)
            val_loss_log[j,:3], val_loss_log[j,3:] = val_loss_epoch, val_spread_epoch
            print("epoch: ",j," train Loss: ",total_epoch_loss_train,"  val Loss", val_loss_epoch, " val spread", val_spread_epoch)
    
    np.savetxt(log_dir+hparams['version']+'.csv', val_loss_log, header = 'val_loss_alpha,val_loss_phic,val_loss_h,val_spread_alpha,val_spread_phic,val_spread_h', delimiter=',', fmt=['%e']*6)
    # save trained bnn
    torch.save({'model' : model.state_dict(), 'guide' : guide}, model_dir+hparams['version']+'_model.pt')
    pyro.get_param_store().save(model_dir+hparams['version']+'_params.pt')
    save_hparam(hparams, hparams['version'])
            
class Rescale:
    '''
    Helper class to pre- and postprocess data
    '''
    def __init__(self,minimum,maximum):
        self.min = minimum
        self.maxmin = maximum - minimum
        
    def to_nn(self,sample_original):
        'Rescale sample from original value to range [0,1] before neural network'
        sample_nn = ((sample_original - self.min) / self.maxmin)
        return sample_nn
    
    def from_nn(self,sample_nn):
        'Rescale sample after neural network back to original range'
        sample_original = sample_nn * self.maxmin + self.min
        return sample_original

class Preprocessor:
    '''
    Rescale labels to range [0,1] for neural network
    '''
    def __init__(self,folder):
        rsc_a = Rescale(bounds_param[0][0],bounds_param[0][1])
        rsc_phi = Rescale(bounds_param[1][0],bounds_param[1][1])
        rsc_h = Rescale(bounds_param[2][0],bounds_param[2][1])

        labels_original = np.loadtxt('data/'+folder+'/labels_raw.csv',delimiter=',',skiprows=1)

        labels_rescaled = np.zeros((labels_original.shape[0],3))
        labels_rescaled[:,0] = rsc_a.to_nn(labels_original[:,0])
        labels_rescaled[:,1] = rsc_phi.to_nn(labels_original[:,1])
        labels_rescaled[:,2] = rsc_h.to_nn(labels_original[:,2])
        np.savetxt('data/'+folder+'/labels_rescaled.csv',labels_rescaled,delimiter=',',fmt=['%f','%f','%f'],header='alpha,phi,h',comments='')
        
class Postprocessor:
    '''
    Rescale parameter outputs of neural networks to original ranges for data assimilation
    '''
    def __init__(self):
        self.rsc_a = Rescale(bounds_param[0][0],bounds_param[0][1])
        self.rsc_phi = Rescale(bounds_param[1][0],bounds_param[1][1])
        self.rsc_h = Rescale(bounds_param[2][0],bounds_param[2][1])
        self.params = np.zeros(3)
        
    def rescale(self, x):
        self.params[0] = (self.rsc_a).from_nn(x[0])
        self.params[1] = (self.rsc_phi).from_nn(x[1])
        self.params[2] = (self.rsc_h).from_nn(x[2])
        return self.params
    
class IdGenerator:
    '''
    Generate lists with random IDs for training and test set
    ''' 
    def generate(self):
        # get random fraction of all available ID's, if all data should be used set fraction=1
        samples_train = int(np.loadtxt('data/background/labels_rescaled.csv',delimiter=',',skiprows=1).shape[0])
        ID_list_train = np.linspace(0,samples_train,samples_train,False,dtype=int)
        random.shuffle(ID_list_train)
        
        samples_val = int(np.loadtxt('data/truth/labels_rescaled.csv',delimiter=',',skiprows=1).reshape((-1,3)).shape[0])
        ID_list_val = np.linspace(0,samples_val,samples_val,False,dtype=int)
        random.shuffle(ID_list_val)            
        return ID_list_train, ID_list_val
        
class Dataset_nn(torch.utils.data.Dataset):
    '''
    Each sample is a full grid on one point in time, shape: (1,750)
    '''
    def __init__(self, ID_list, folder, hparams):
        self.ID_list = ID_list
        self.folder = folder
        self.labels = np.loadtxt('data/'+folder+'/labels_rescaled.csv',delimiter=',',skiprows=1)
        if folder == 'truth':
            if hparams['obs']:
                self.input_folder = 'obs'
            else:
                self.input_folder = 'analysis'
        else:
            self.input_folder = folder
        if hparams['obs']:
            self.H = np.loadtxt('data/H.csv',delimiter=',',dtype='int')
        else:
            self.H = np.arange(750)
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ID_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.ID_list[index]
        X = np.loadtxt('data/'+self.input_folder+'/'+str(ID)+'.csv',delimiter=',')[...,self.H]
        Y = self.labels[ID]
        
        return X,Y
    
class DataModule():
    def __init__(self, hparams):
        
        self.hparams = hparams
    
        ID_train, ID_val = IdGenerator().generate()
        
        # get Datasets
        train_dataset = Dataset_nn(ID_train,'background',hparams)
        val_dataset = Dataset_nn(ID_val,'truth',hparams)

        # assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
    def get_data(self):
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset,shuffle=True,batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'],drop_last = False)
        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.hparams['batch_size_val'], num_workers=self.hparams['num_workers'], drop_last = False)
        return train_dataloader, val_dataloader
    
class BNN_fc(PyroModule):
    def __init__(self,hparams):
        super().__init__()
        if hparams['obs']:
            self.h0 = len(np.loadtxt('data/H.csv',delimiter=',',dtype='int'))
        else:
            self.h0 = 750
        h = hparams['hidden_layers']
        self.hparams = hparams
        
        self.fc1 = PyroModule[nn.Linear](self.h0, h[0])
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([h[0], self.h0]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([h[0]]).to_event(1))
        
        self.fc2 = PyroModule[nn.Linear](h[0], h[1])
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([h[1], h[0]]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([h[1]]).to_event(1))
        
        self.fc3 = PyroModule[nn.Linear](h[1], h[2])
        self.fc3.weight = PyroSample(dist.Normal(0., 1.).expand([h[2], h[1]]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(0., 1.).expand([h[2]]).to_event(1))
        
        self.fc4 = PyroModule[nn.Linear](h[2], 3)
        self.fc4.weight = PyroSample(dist.Normal(0., 1.).expand([3, h[2]]).to_event(2))
        self.fc4.bias = PyroSample(dist.Normal(0., 1.).expand([3]).to_event(1))
        
        self.relu = nn.LeakyReLU()
        self.batchnorm = nn.BatchNorm1d(self.h0)
        
    def forward(self, x, y=None):
        x = torch.flatten(x, 1)
        x = self.batchnorm(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        mu = self.fc4(x).squeeze()
        sigma =  pyro.sample("sigma", dist.Uniform(self.hparams['sigma']-0.1*self.hparams['sigma'], self.hparams['sigma']+0.1*self.hparams['sigma']))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)
            return obs
    
def val_loss(model,guide,val_loader,hparams,samples,nens):
    '''
    validation loss during training
    '''
    predictive = Predictive(model, guide=guide, num_samples=samples)
    val_size = len(val_loader)*nens
    label_mean = np.zeros((len(val_loader), 3))
    label_val = np.zeros((len(val_loader), 3))
    label_spread_mean = np.zeros((len(val_loader), 3))

    for batch_id, data in enumerate(val_loader):
        x = data[0][0,...]
        label_spread = (predictive(x.float())['obs'].detach().numpy()).reshape((-1,3))
        label_mean[batch_id,...] = np.mean(label_spread,axis=0)
        label_val[batch_id,...] = data[1][...]
        label_spread_mean[batch_id,...] = np.std(label_spread,axis=0)
        
    rmse = np.mean((label_mean-label_val)**2,axis=0)**(1/2)
    spread = np.mean(label_spread_mean,axis=0)
    
    return rmse, spread, label_mean, label_val

def DA(mode,DA_cycles_total,exp_ID_start,exp_ID_end,nens,samples,n_train):
    '''
    Full data assimilation experiment for one ground truth parameter set.
    mode : 'true','false','bnn'
    DA_cycles : number of data assimilation cycles
    exp_ID_start, exp_ID_end : start and end IDs of ground truth parameters
    nens : analysis/background ensemble members
    samples : number of samples the bnn draws for each input
    '''
    for exp_ID in range(exp_ID_start,exp_ID_end):
        if mode in ['true','false']:
            DataGenerator_const_init(nens=nens,exp_ID=exp_ID,mode=mode)
        else:
            DataGenerator_ml_init(ml_model=mode,nens=nens,samples=samples,exp_ID=exp_ID,n_train=n_train)
        rmse_spread_params_vars = np.zeros((2,6))
        for cycle in range(DA_cycles_total):
            DataGenerator_const(nens=nens,DA_cycle=cycle,mode=mode)

            params_bnn = np.loadtxt('data/online/background/labels_rescaled.csv',delimiter=',',skiprows=1)
            params_truth = (np.loadtxt('data/truth/labels_rescaled.csv',delimiter=',',skiprows=1)[exp_ID,:])

            rmse_spread_params_vars[0,:3] = (np.mean(params_bnn,axis=0)-params_truth)**2
            rmse_spread_params_vars[0,3:] = np.std(params_bnn,axis=0)
            analysis = np.zeros((750,nens))
            for i in range(nens):
                analysis[:,i] = np.loadtxt('data/online/analysis/'+str(i)+'.csv',delimiter=',')
            truth = np.loadtxt('data/online/truth/'+str(0)+'.csv',delimiter=',')
            for j in range(3):
                rmse_spread_params_vars[1,j] = np.mean((np.mean(analysis[250*j:250*(j+1),:],axis=1)-truth[250*j:250*(j+1)])**2)**(1/2)
                rmse_spread_params_vars[1,j+3] = np.mean(np.std(analysis[250*j:250*(j+1),:],axis=1))
            create_folder_path('data/online/metrics/'+mode+'/'+str(nens)+'/'+str(exp_ID))
            if cycle >= 50:
                np.savetxt('data/online/metrics/'+mode+'/'+str(nens)+'/'+str(exp_ID)+'/'+str(cycle)+'.csv', rmse_spread_params_vars, header = 'rmse alpha/u,rmse phic/h,rmse h/r,spread alpha/u,spread phic/h,spread h/r', delimiter=',', fmt=['%e']*6)
            
def save_parameter_preds_bnn(nens,n_train,exp_ID_start,exp_ID_end,samples,val_obs):
    '''
    Saves the parameter predictions + ground truths for each analysis ensemble member in 'data/parameters/<nens>_nens/<n_train>_ntrain/<exp_ID>/' for 
    visualisation/calculation of metrics/etc....
    '''
    create_folder_path('data/parameters_bnn')
    for exp_ID in range(exp_ID_start,exp_ID_end):
        version = str(nens)
        hparams = load_hparam(version)
        model = BNN_fc(hparams)
        saved_model_dict = torch.load('models/'+version+'_model.pt')
        saved_guide_dict = torch.load('models/'+version+'_params.pt')

        model.load_state_dict(saved_model_dict['model'])
        guide = saved_model_dict['guide']
        pyro.get_param_store().load('models/'+version+'_params.pt')
        model.eval()
        create_folder_path('data/parameters_bnn/'+str(nens)+'_nens/'+str(n_train)+'_ntrain/'+str(exp_ID))

        predictive = Predictive(model, guide=guide, num_samples=samples)
        
        if val_obs:
            H = np.loadtxt('data/H.csv',delimiter=',',dtype='int')
            x = np.loadtxt('data/obs/'+str(exp_ID)+'.csv',delimiter=',')[0,H].reshape((1,-1))
            x = torch.from_numpy(x)
            y = predictive(x.float())['obs'].detach().numpy().squeeze()
            np.savetxt('data/parameters_bnn/'+str(nens)+'_nens/'+str(n_train)+'_ntrain/'+str(exp_ID)+'/output.csv',y[:,:],delimiter=',',fmt=['%f']*3,header='alpha,phi,h',comments='')
            
        else:
            x = np.loadtxt('data/analysis/'+str(exp_ID)+'.csv',delimiter=',')
            x = torch.from_numpy(x)

            y = predictive(x.float())['obs'].detach().numpy().squeeze()
            for i in range(nens):
                np.savetxt('data/parameters_bnn/'+str(nens)+'_nens/'+str(n_train)+'_ntrain/'+str(exp_ID)+'/'+str(i)+'_output.csv',y[:,i,:],delimiter=',',fmt=['%f']*3,header='alpha,phi,h',comments='')
        
        np.savetxt('data/parameters_bnn/'+str(nens)+'_nens/'+str(n_train)+'_ntrain/'+str(exp_ID)+'/input.csv',x,delimiter=',',fmt=['%f']*x.shape[1],comments='')

        # truth parameters
        y_truth = np.loadtxt('data/truth/labels_rescaled.csv',delimiter=',',skiprows=1)[exp_ID,:]
        np.savetxt('data/parameters_bnn/'+str(nens)+'_nens/'+str(n_train)+'_ntrain/'+str(exp_ID)+'/truth_output.csv',y_truth.reshape((1,3)),delimiter=',',fmt=['%f']*3,header='alpha,phi,h',comments='')
        
def save_parameter_preds_rf(nens,n_train,exp_ID_start,exp_ID_end,val_obs):
    '''
    Saves the parameter predictions + ground truths for each analysis ensemble member in 'data/parameters/<nens>_nens/<n_train>_ntrain/<exp_ID>/' for 
    visualisation/calculation of metrics/etc....
    '''
    
    for exp_ID in range(exp_ID_start,exp_ID_end):
        version = str(nens)
        create_folder_path('data/parameters_rf/'+str(nens)+'_nens/'+str(n_train)+'_ntrain/'+str(exp_ID))
        model = pickle.load(open("models/"+version+"rf", 'rb')) # load random forest model
        
        if val_obs:
            H = np.loadtxt('data/H.csv',delimiter=',',dtype='int')
            x = (np.loadtxt('data/obs/'+str(exp_ID)+'.csv',delimiter=',')[0,H]).reshape((1,-1))
            y = np.array([tree.predict(x) for tree in model])
        
            np.savetxt('data/parameters_rf/'+str(nens)+'_nens/'+str(n_train)+'_ntrain/'+str(exp_ID)+'/output.csv',y[:,0,:],delimiter=',',fmt=['%f']*3,header='alpha,phi,h',comments='')
        else:
            x = np.loadtxt('data/analysis/'+str(exp_ID)+'.csv',delimiter=',')
            y = np.array([tree.predict(x) for tree in model])
            for i in range(nens):
                np.savetxt('data/parameters_rf/'+str(nens)+'_nens/'+str(n_train)+'_ntrain/'+str(exp_ID)+'/'+str(i)+'_output.csv',y[:,i,:],delimiter=',',fmt=['%f']*3,header='alpha,phi,h',comments='')
        
        np.savetxt('data/parameters_rf/'+str(nens)+'_nens/'+str(n_train)+'_ntrain/'+str(exp_ID)+'/input.csv',x,delimiter=',',fmt=['%f']*x.shape[1],comments='')

        # truth parameters
        y_truth = np.loadtxt('data/truth/labels_rescaled.csv',delimiter=',',skiprows=1)[exp_ID,:]
        np.savetxt('data/parameters_rf/'+str(nens)+'_nens/'+str(n_train)+'_ntrain/'+str(exp_ID)+'/truth_output.csv',y_truth.reshape((1,3)),delimiter=',',fmt=['%f']*3,header='alpha,phi,h',comments='')
        
def plot_parameters(ml_model,nens,n_train,n_val,samples,val_obs):
    '''
    Plots mean and standard deviation of all <n_val> ground truth parameters using bnn trained with nens analysis ensemble members ans n_train
    training samples.
    '''
    params_truth = np.zeros((3,n_val))
    if val_obs:
        params_ml = np.zeros((3,n_val,samples))
    else:
        params_ml = np.zeros((3,n_val,nens,samples))
    for i in range(n_val):
        params_truth[:,i] = np.loadtxt('data/parameters_'+ml_model+'/'+str(nens)+'_nens/'+str(n_train)+'_ntrain/'+str(i)+'/truth_output.csv',delimiter=',',skiprows=1)
        if val_obs:
            params_ml[:,i,...] = np.loadtxt('data/parameters_'+ml_model+'/'+str(nens)+'_nens/'+str(n_train)+'_ntrain/'+str(i)+'/output.csv',delimiter=',',skiprows=1).T
        else:
            for j in range(nens):
                params_ml[:,i,j,...] = np.loadtxt('data/parameters_'+ml_model+'/'+str(nens)+'_nens/'+str(n_train)+'_ntrain/'+str(i)+'/'+str(j)+'_output.csv',delimiter=',',skiprows=1).T

    params_ml_mean = np.mean(params_ml.reshape((3,n_val,-1)),axis=2)
    params_ml_std = np.std(params_ml.reshape((3,n_val,-1)),axis=2)
    
    fig, ax = plt.subplots(1,3,figsize = (20,6))

    params_names = ['alpha','phi','hr']
    ylabels = [ml_model+' prediction','','']

    rmse = np.mean((params_ml_mean-params_truth)**2,axis=1)**(1/2)
    spread = np.mean(params_ml_std,axis=1)

    for i in range(3):
        ax[i].errorbar(x=params_truth[i,:],y=params_ml_mean[i,:],yerr=params_ml_std[i,:],fmt='.',color='gray')
        ax[i].plot(params_truth[i,:],params_truth[i,:],'-r',label='perfect prediction')
        ax[i].set_title(params_names[i]+' (RMSE:'+str(round(rmse[i],2))+', spread:'+str(round(spread[i],2))+')')
        ax[i].set_ylabel(ylabels[i])
        ax[i].set_xlabel('ground truth')
    plt.legend()
    return rmse,spread
    
def plot_parameters_hist(ml_model,exp_ID,nens,n_train,samples,val_obs):
    '''
    Plots histogramm of parameter predictions of ground truth parameters related to <exp_ID> using bnn trained with nens analysis ensemble members ans n_train
    training samples. Total number of parameter predictions = nens*samples
    '''
    fig, ax = plt.subplots(1,3,figsize = (20,6))
    if val_obs:
        params_ml = np.zeros((3,samples))
        params_ml = np.loadtxt('data/parameters_'+ml_model+'/'+str(nens)+'_nens/'+str(n_train)+'_ntrain/'+str(exp_ID)+'/output.csv',delimiter=',',skiprows=1).T
    else:
        params_ml = np.zeros((3,nens,samples))
        for j in range(nens):
            params_ml[:,j,:] = np.loadtxt('data/parameters_'+ml_model+'/'+str(nens)+'_nens/'+str(n_train)+'_ntrain/'+str(exp_ID)+'/'+str(j)+'_output.csv',delimiter=',',skiprows=1).T
        
    params_truth = np.loadtxt('data/parameters_'+ml_model+'/'+str(nens)+'_nens/'+str(n_train)+'_ntrain/'+str(exp_ID)+'/truth_output.csv',delimiter=',',skiprows=1)

    params_ml = params_ml.reshape((3,-1))
    params_names = ['alpha','phi','hr']
    ylabels = ['','','']

    for i in range(3):
        sn.histplot(params_ml[i,:],ax=ax[i],label=ml_model)
        ax[i].axvline(params_truth[i],color='red',label='truth')
        ax[i].set_xlim(0, 1)
        ax[i].set_title(params_names[i])
        ax[i].set_ylabel(ylabels[i])
    plt.legend(fontsize=15)
