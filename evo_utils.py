from parameters import *
import os, sys, time, pickle
import itertools

import numpy as np
if len(sys.argv) > 1:
    import cupy as cp
    cp.cuda.Device(sys.argv[1]).use()
else:
    import numpy as cp


### GPU utilities
def to_gpu(x):
    """ Move numpy arrays (or dicts of arrays) to GPU """
    if type(x) == dict:
        return {k:cp.asarray(a) for (k, a) in x.items()}
    else:
        return cp.asarray(x)

def to_cpu(x):
    """ Move cupy arrays (or dicts of arrays) to CPU """
    if len(sys.argv) > 1:
        if type(x) == dict:
            return {k:cp.asnumpy(a) for (k, a) in x.items()}
        else:
            return cp.asnumpy(x)
    else:
        if type(x) == dict:
            return {k:a for (k, a) in x.items()}
        else:
            return x


### Network functions

def relu(x):
    """ Performs relu on x """
    return cp.maximum(0., x, dtype=x.dtype)

def pad(x):
    shape = x.shape
    idx = shape[2]-1 #127 for 128x128 img
    temp = cp.zeros((par['n_networks'],par['batch_train_size'],130,130,shape[-1]))
    temp[:,:,1:idx+2,1:idx+2,:] = x

    temp[:,:,0,0,:]         = x[:,:,0,0,:]
    temp[:,:,0,idx+2,:]       = x[:,:,0,idx,:]
    temp[:,:,idx+2,0,:]       = x[:,:,idx,0,:]
    temp[:,:,idx+2,idx+2,:]     = x[:,:,idx,idx,:]
    
    temp[:,:,0,1:idx+2,:]     = x[:,:,0,:,:]
    temp[:,:,idx+2,1:idx+2,:]   = x[:,:,idx,:,:]
    temp[:,:,1:idx+2,0,:]     = x[:,:,:,0,:]
    temp[:,:,1:idx+2,idx+2,:]   = x[:,:,:,idx,:]
    return temp

def apply_filter(x, filt):
    temp = cp.zeros((par['n_networks'],par['batch_train_size'],128,128))
    for i in range(128):
        for j in range(128):
            temp[:,:,i,j] = cp.sum(x[:,:,i:i+3,j:j+3,:] * cp.repeat(cp.expand_dims(filt,axis=1),par['batch_train_size'],axis=1),axis=(2,3,4))

    return temp

def convolve(x, var_dict, filt_type):
    if filt_type == 'conv1_filter':
        conv = cp.zeros((par['n_networks'],par['batch_train_size'],*par['inp_img_shape'],par['num_conv1_filters']))
    else:
        conv = cp.zeros((par['n_networks'],par['batch_train_size'],*par['inp_img_shape'],3))
    
    i = 0
    x = pad(x)
    for key in var_dict.keys():
        if filt_type in key:
            conv[:,:,:,:,i] = apply_filter(x, var_dict[key])
            i += 1

    return conv

### Optimization functions

def cross(var1, var2, rate):
    """ Transmit some of var2 over to var1, based on the give rate """
    return cp.where(cp.random.choice([True,False], size=var1.shape, p=[1-rate, rate]), var1, var2)

def mutate(var, num, rate, scale, epsilon=0.):
    """ Mutates a given variable by a given rate and scale,
        generating as many offspring as num """
    mutation_mask = cp.random.choice([1,0], [num, *var.shape], p=[rate, 1-rate]).astype(cp.float32)
    mutation = cp.random.normal(loc=epsilon, scale=scale, size=[num, *var.shape])
    return cp.repeat(cp.expand_dims(var,axis=0),num,axis=0) + mutation*mutation_mask

