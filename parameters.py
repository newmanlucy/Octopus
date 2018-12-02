import numpy as np
import os

print("Loading parameters...")

"""
Set independent parameters
"""
par = {
    # Setup parameters
    'save_dir'          : './savedir/',
    'input_dir'         : './bw_im2/',
    'target_dir'        : './raw_im2/',
    'simulation'	    : False,
    'img_size'          : 128,

    # Network shape
    'n_enc'             : 125,
    'n_link'            : 100,
    'n_latent'          : 75,
    'n_dec'             : 125,
    'num_layers'        : 3,
    'new_model'         : False,
    'num_conv1_filters' : 32,

    # Evolutionary
    'n_networks'        : 100,
    'survival_rate'     : 0.1,
    'mutation_rate'     : 0.1,
    'mutation_strength' : 0.20,
    'cross_rate'        : 0,
    'use_crossing'      : False,
    
    # Training
    'task'              : 'bw_to_bw_simple',
    'learning_rate'     : 0.001,
    'connection_prob'   : 1,
    'dropout'           : 1,

    # Variance
    'input_mean'        : 0.0,
    'noise_in_sd'       : 0.1,
    'noise_rnn_sd'      : 0.5,

    # Training setup
    'normalize01'       : False,
    'batch_train_size'  : 32,
    'num_iterations'    : 30001,
    'print_iter'        : 10,
    'save_iter'         : 2000
}

"""
Set dependent parameters
"""
def update_dependencies():
    print('Updating dependencies...\n')

    if par['task'] == 'bw_to_bw':
        par['input_dir'] = './bw_im'
        par['target_dir'] = './bw_im'
        par['inp_img_shape'] = (par['img_size'],par['img_size'])    # used to be (par['img_size'],par['img_size'],3)
        par['out_img_shape'] = (par['img_size'],par['img_size'],3)
        par['n_input'] = par['img_size']*par['img_size'] #par['img_size']*par['img_size']*3
        par['n_output'] = par['n_input']*3 # par['n_input']
        par['save_dir'] = './savedir/bw_to_bw/'

    elif par['task'] == 'bw_to_bw_simple':
        par['input_dir'] = './bw_im'
        par['target_dir'] = './bw_im'
        par['inp_img_shape'] = (par['img_size'],par['img_size'])
        par['out_img_shape'] = (par['img_size'],par['img_size'])
        par['n_input'] = par['img_size']*par['img_size']
        par['n_output'] = par['n_input']
        par['save_dir'] = './savedir/bw_to_bw_simple/'

    elif par['task'] == 'bw3_to_color':
        par['input_dir'] = './bw_im/'
        par['target_dir'] = './raw_im/'
        par['inp_img_shape'] = (par['img_size'],par['img_size'],3)
        par['out_img_shape'] = (par['img_size'],par['img_size'],3)
        par['n_input'] = par['img_size']*par['img_size']*3
        par['n_output'] = par['n_input']
        par['save_dir'] = './savedir/bw3_to_color/'

    elif par['task'] == 'bw1_to_color':
        par['input_dir'] = './bw_im/'
        par['target_dir'] = './raw_im/'
        par['inp_img_shape'] = (par['img_size'],par['img_size'])
        par['out_img_shape'] = (par['img_size'],par['img_size'],3)
        par['n_input'] = par['img_size']*par['img_size']
        par['n_output'] = par['n_input']*3
        par['save_dir'] = './savedir/bw1_to_color/'

    elif par['task'] == 'conv_task':
        par['input_dir'] = './bw_im/'
        par['target_dir'] = './raw_im/'
        par['inp_img_shape'] = (par['img_size'],par['img_size'])
        par['out_img_shape'] = (par['img_size'],par['img_size'],3)
        par['n_input'] = par['img_size']*par['img_size']
        par['n_output'] = par['n_input']*3
        par['save_dir'] = './savedir/conv_task/'

    if par['simulation']:
        par['save_dir'] = './simulation/'

    par['num_survivors'] = int(par['n_networks'] * par['survival_rate'])

"""
Update parameters based on the given dictionary (updates)
"""
def update_parameters(updates):
    print('Updating parameters...')

    for (key, val) in updates.items():
        par[key] = val

    # Update any dependent parameters
    update_dependencies()


# Run update dependencies when loading parameters
update_dependencies()
