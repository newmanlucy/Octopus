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

def matmul(a, b):
    """ Does matrix multiplication as a @ b, accounting for
        whether either is of dtype=int8 """
    if cp.int8 in [a.dtype, b.dtype]:
        # Set up for a=state, b=weights
        return cp.sum(a[...,cp.newaxis]*b[:,cp.newaxis,...], axis=-2, dtype=cp.float16)
    else:
        return cp.matmul(a, b)

def relu(x):
    """ Performs relu on x """
    return cp.maximum(0., x, dtype=x.dtype)

def softmax(x, a=-1):
    """ Performs stable softmax on x, across the last axis by default """
    c = cp.exp(x-cp.amax(x, axis=a, keepdims=True))
    return c/cp.sum(c, axis=a, keepdims=True).astype(cp.float32)

def pad(x):
    shape = x.shape
    temp = cp.zeros((par['n_networks'],par['batch_train_size'],130,130,shape[-1]))
    temp[:,:,1:129,1:129,:] = x

    temp[:,:,0,0,:]         = x[:,:,0,0,:]
    temp[:,:,0,129,:]       = x[:,:,0,127,:]
    temp[:,:,129,0,:]       = x[:,:,127,0,:]
    temp[:,:,129,129,:]     = x[:,:,127,127,:]
    
    temp[:,:,0,1:129,:]     = x[:,:,0,:,:]
    temp[:,:,129,1:129,:]   = x[:,:,127,:,:]
    temp[:,:,1:129,0,:]     = x[:,:,:,0,:]
    temp[:,:,1:129,129,:]   = x[:,:,:,127,:]
    return temp

def apply_filter(x, filt):
    temp = cp.zeros((par['n_networks'],par['batch_train_size'],128,128))
    for i in range(128):
        for j in range(128):
            temp[:,:,i,j] = np.sum(x[:,:,i:i+3,j:j+3,:] * np.expand_dims(filt,axis=1))

    return temp

def convolve(x, var_dict, filt_type):
    filters = []
    for key in var_dict.keys():
        if filt_type in key:
            filters.append(key)

    input_shape = x.shape
    filt_shape = var_dict[filters[0]].shape

    if filt_type == 'conv1_filter':
        conv = cp.zeros((par['n_networks'], par['batch_train_size'],*par['inp_img_shape'],par['num_conv1_filters']))
    else:
        conv = cp.zeros((par['n_networks'], par['batch_train_size'],*par['inp_img_shape'],3))
    
    x = pad(x)
    for i in range(len(filters)):
        conv[:,:,:,:,i] = apply_filter(x, var_dict[filters[i]])

    return conv


### Judgement functions

def cross_entropy(mask, target, output, eps=1e-16):
    """ Calculate the cross entropy loss for a rate-based network """
    mask   = mask.astype(cp.float32)
    target = target.astype(cp.float32)
    output = output.astype(cp.float32)
    return -cp.mean(mask[...,cp.newaxis]*target*cp.log(softmax(output)+eps), axis=(0,2,3)).astype(cp.float32)


### Optimization functions

def cross(var1, var2, rate):
    """ Transmit some of var2 over to var1, based on the give rate """
    return cp.where(cp.random.choice([True,False], size=var1.shape, p=[rate, 1-rate]), var1, var2)

def mutate(var, num, rate, scale, epsilon=0.):
    """ Mutates a given variable by a given rate and scale,
        generating as many offspring as num """
    #mutation_mask = cp.random.random(size=[num, *var.shape], dtype=np.float32).astype(cp.float32)
    mutation_mask = cp.random.random(size=[num, *var.shape]).astype(cp.float32)
    mutation = cp.random.normal(loc=epsilon, scale=scale, size=[num, *var.shape])
    return var[cp.newaxis,...] + mutation*mutation_mask

