import numpy as np
import os

print("Loading parameters...")

"""
Set independent parameters
"""
par = {
    # Setup parameters
    'save_dir'          : './savedir/',
    'input_dir'         : './bw_im/',
    'target_dir'        : './raw_im/',
    'img_size'          : 128,

    # Network shape
    'n_enc'             : 125,
    'n_link'            : 100,
    'n_latent'          : 75,
    'n_dec'             : 125,
    'num_layers'        : 3,
    'new_model'         : False,
    
    # Training
    'task'              : 'bw_to_bw_simple',
    'learning_rate'     : 0.1,
    'connection_prob'   : 1,

    # Variance
    'input_mean'        : 0.0,
    'noise_in_sd'       : 0.1,
    'noise_rnn_sd'      : 0.5,

    # Training setup
    'batch_train_size'  : 100,
    'num_iterations'    : 600001,
    'print_iter'        : 1000,
    'save_iter'         : 10000
}

"""
Set dependent parameters
"""
def update_dependencies():
    print('Updating dependencies...\n')

    if par['task'] == 'bw_to_bw':
        par['input_dir'] = './bw_im'
        par['target_dir'] = './bw_im'
        par['img_shape'] = (par['img_size'],par['img_size'],3)
        par['n_input'] = par['img_size']*par['img_size']*3
        par['n_output'] = par['n_input']
        par['savedir'] = './savedir/bw_to_bw'

    elif par['task'] == 'bw_to_bw_simple':
        par['input_dir'] = './bw_im'
        par['target_dir'] = './bw_im'
        par['img_shape'] = (par['img_size'],par['img_size'])
        par['n_input'] = par['img_size']*par['img_size']
        par['n_output'] = par['n_input']
        par['savedir'] = './savedir/bw_to_bw_simple'

    elif par['task'] == 'bw3_to_color':
        par['input_dir'] = './bw_im'
        par['target_dir'] = './raw_im'
        par['img_shape'] = (par['img_size'],par['img_size'],3)
        par['n_input'] = par['img_size']*par['img_size']*3
        par['n_output'] = par['n_input']
        par['savedir'] = './savedir/bw3_to_color'

    # Set up initializers
    if par['num_layers'] == 5:
        par['W_in_init'] = np.float32(np.random.normal(size=[par['n_input'], par['n_enc']]))
        par['b_enc_init'] = np.float32(np.random.normal(size=(1,par['n_enc'])))

        par['W_enc_init'] = np.float32(np.random.normal(size=[par['n_enc'], par['n_link']]))
        par['b_latent_init'] = np.float32(np.random.normal(size=(1,par['n_link'])))

        par['W_link_init'] = np.float32(np.random.normal(size=[par['n_link'], par['n_latent']]))
        par['b_link_init'] = np.float32(np.random.normal(size=(1,par['n_latent'])))
        
        par['W_dec_init'] = np.float32(np.random.normal(size = [par['n_latent'], par['n_link']]))
        par['b_dec_init'] = np.float32(np.random.normal(size=(1,par['n_link'])))

        par['W_link2_init'] = np.float32(np.random.normal(size=[par['n_link'], par['n_dec']]))
        par['b_link2_init'] = np.float32(np.random.normal(size=(1,par['n_dec'])))

        par['W_out_init'] = np.float32(np.random.normal(size=[par['n_dec'], par['n_output']]))
        par['b_out_init'] = np.float32(np.random.normal(size=(1,par['n_output'])))
    
    elif par['num_layers'] == 3:
        par['W_in_init'] = np.float32(np.random.normal(size=[par['n_input'], par['n_enc']]))
        par['b_enc_init'] = np.float32(np.random.normal(size=(1,par['n_enc'])))

        par['W_enc_init'] = np.float32(np.random.normal(size=[par['n_enc'], par['n_latent']]))
        par['b_latent_init'] = np.float32(np.random.normal(size=(1,par['n_latent'])))
        
        par['W_dec_init'] = np.float32(np.random.normal(size=[par['n_latent'], par['n_dec']]))
        par['b_dec_init'] = np.float32(np.random.normal(size=(1,par['n_dec'])))

        par['W_out_init'] = np.float32(np.random.normal(size=[par['n_dec'], par['n_output']]))
        par['b_out_init'] = np.float32(np.random.normal(size=(1,par['n_output'])))
    
    elif par['num_layers'] == 2:
        par['W_in_init'] = np.float32(np.random.normal(size=[par['n_input'], par['n_enc']]))
        par['b_enc_init'] = np.float32(np.random.normal(size=(1,par['n_enc'])))

        par['W_dec_init'] = np.float32(np.random.normal(size=[par['n_enc'], par['n_dec']]))
        par['b_dec_init'] = np.float32(np.random.normal(size=(1,par['n_dec'])))

        par['W_out_init'] = np.float32(np.random.normal(size=[par['n_dec'], par['n_output']]))
        par['b_out_init'] = np.float32(np.random.normal(size=(1,par['n_output'])))

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