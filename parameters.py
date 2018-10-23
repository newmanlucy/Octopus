import numpy as np
import os

print("Loading parameters...")

"""
Set independent parameters
"""
par = {
	# Setup parameters
	'save_dir'			: './savedir/',
	'img_dir'			: './bw_im/',
	'img_size'			: 32,

	# Network shape
	'n_enc'				: 125,
	'n_link'			: 100,
	'n_latent'			: 75,
	'n_dec'				: 125,
	
	# Training
	'task'				: 'bw_to_bw_simple',
	'learning_rate'		: 5e-2,
	'connection_prob'	: 1,

	# Variance
	'input_mean'		: 0.0,
	'noise_in_sd'		: 0.1,
	'noise_rnn_sd'		: 0.5,

	# Training setup
	'batch_train_size'	: 256,
	'num_iterations'	: 50001,
	'print_iter'		: 100,
}

"""
Set dependent parameters
"""
def update_dependencies():
	print('Updating dependencies...\n')

	if par['task'] == 'bw_to_bw':
		par['data_dir'] = './bw_im'
		par['img_shape'] = (par['img_size'],par['img_size'],3)
		par['n_input'] = par['img_size']*par['img_size']*3
		par['n_output'] = par['n_input']

	elif par['task'] == 'bw_to_bw_simple':
		par['data_dir'] = './bw_im'
		par['img_shape'] = (par['img_size'],par['img_size'])
		par['n_input'] = par['img_size']*par['img_size']
		par['n_output'] = par['n_input']

	# Set up initializers
	c = 0.1
	par['W_in_init'] = c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_input'], par['n_enc']]))
	
	par['W_enc_init'] = c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_enc'], par['n_link']]))
	par['b_latent_init'] = np.zeros((1,par['n_link']), dtype = np.float32)

	par['W_link_init'] = c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_link'], par['n_latent']]))
	par['b_link_init'] = np.zeros((1,par['n_latent']), dtype = np.float32)
	
	par['W_dec_init'] = c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_latent'], par['n_link']]))
	par['b_dec_init'] = np.zeros((1,par['n_link']), dtype = np.float32)

	par['W_link2_init'] = c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_link'], par['n_dec']]))
	par['b_link2_init'] = np.zeros((1,par['n_dec']), dtype = np.float32)

	par['W_out_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_dec'], par['n_output']]))
	par['b_out_init'] = np.zeros((1,par['n_output']), dtype = np.float32)

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