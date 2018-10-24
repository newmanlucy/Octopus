import json
import time
import model
import sys
import io
from parameters import *
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

def try_model(updates):

    # GPU
    try:
        gpu_id = sys.argv[1]
    except:
        gpu_id = None

    # Update parameters for this run
    print('Setting up model!')
    update_parameters(updates)

    # Save updated parameters
    with io.open(par['save_dir']+'run_'+str(par['run_number'])+'_params.json', 'w', encoding='utf8') as outfile:
        str_ = json.dumps(updates,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))

    # Run Model
    t0 = time.time()
    try:
        print('Model number ' + str(par['run_number']) + ' running!')
        model.main(gpu_id)
        print('Model run concluded.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))

# updates = {
#     'img_size'          : 64,
#     'n_enc'             : 125,
#     'n_latent'          : 75,
#     'n_dec'             : 125,
#     'run_number'        : 0,
#     'print_iter'        : 1000,
# }
# try_model(updates)

# updates = {
#     'img_size'          : 64,
#     'n_enc'             : 300,
#     'n_latent'          : 75,
#     'n_dec'             : 300,
#     'run_number'        : 1,
#     'print_iter'        : 1000,
# }
# try_model(updates)

# updates = {
#     'img_size'          : 128,
#     'n_enc'             : 125,
#     'n_latent'          : 75,
#     'n_dec'             : 125,
#     'run_number'        : 6,
#     'print_iter'        : 1000,
#     'batch_train_size'  : 5,
#     'num_iterations'    : 25001,
# }
# try_model(updates)

# updates = {
#     'img_size'          : 128,
#     'n_enc'             : 300,
#     'n_latent'          : 75,
#     'n_dec'             : 300,
#     'run_number'        : 3,
#     'print_iter'        : 1000,
# }
# try_model(updates)

# updates = {
#     'img_size'          : 128,
#     'n_enc'             : 300,
#     'n_latent'          : 125,
#     'n_dec'             : 300,
#     'run_number'        : 4,
#     'print_iter'        : 1000,
# }
# try_model(updates)

# updates = {
#     'task'              : 'bw_to_bw_simple',
#     'save_dir'          : './savedir/bw_to_bw_simple/',
#     'img_size'          : 128,
#     'n_enc'             : 500,
#     'n_latent'          : 100,
#     'n_dec'             : 500,
#     'run_number'        : 0,
#     'print_iter'        : 1000,
#     'batch_train_size'  : 1,
#     'num_iterations'    : 30001,
# }
# try_model(updates)

# Literature says feed-forward autoencoder
# works for images of 28x28 with neurons 784->32->32->784
# updates = {
#     'task'              : 'bw_to_bw_simple',
#     'save_dir'          : './savedir/bw_to_bw_simple/',
#     'img_size'          : 28,
#     'n_enc'             : 500,
#     'n_latent'          : 32,
#     'n_dec'             : 500,
#     'run_number'        : 1,
#     'print_iter'        : 1000,
#     'save_iter'         : 10000,
#     'batch_train_size'  : 100,
#     'num_iterations'    : 600001,
# }
# try_model(updates)

# Shitty normalizing version with 128 x 128 on literature model
# updates = {
#     'task'              : 'bw_to_bw_simple',
#     'save_dir'          : './savedir/bw_to_bw_simple/',
#     'use_literature_code' : True,
#     'img_size'          : 128,
#     'n_enc'             : 500,
#     'n_latent'          : 32,
#     'n_dec'             : 500,
#     'run_number'        : 2,
#     'print_iter'        : 1000,
#     'save_iter'         : 10000,
#     'batch_train_size'  : 100,
#     'num_iterations'    : 600001,
# }
# try_model(updates)

# Shitty normalizing version on 5 LAYERs our model
# updates = {
#     'task'              : 'bw_to_bw_simple',
#     'save_dir'          : './savedir/bw_to_bw_simple/',
#     'use_literature_code' : False,
#     'img_size'          : 128,
#     'n_enc'             : 500,
#     'n_link'            : 100,
#     'n_latent'          : 75,
#     'n_dec'             : 500,
#     'run_number'        : 3,
#     'print_iter'        : 1000,
#     'save_iter'         : 10000,
#     'batch_train_size'  : 100,
#     'num_iterations'    : 600001,
# }
# try_model(updates)

# # GOOD normalization, literature code
# updates = {
#     'task'              : 'bw_to_bw_simple',
#     'save_dir'          : './savedir/bw_to_bw_simple/',
#     'normalize'         : True,
#     'new_model'         : False,
#     'use_literature_code' : True,
#     'img_size'          : 128,
#     'n_enc'             : 500,
#     'n_latent'          : 32,
#     'n_dec'             : 500,
#     'run_number'        : 4,
#     'print_iter'        : 1000,
#     'save_iter'         : 1000,
#     'batch_train_size'  : 100,
#     'num_iterations'    : 600001,
# }
# try_model(updates)

# # Shitty normalizing version with 128 x 128 on literature model
# # with doubled the size of the  latent layer
# updates = {
#     'task'              : 'bw_to_bw_simple',
#     'save_dir'          : './savedir/bw_to_bw_simple/',
#     'use_literature_code' : True,
#     'new_model'         : False,
#     'normalize'         : False,
#     'img_size'          : 128,
#     'n_enc'             : 500,
#     'n_latent'          : 64,
#     'n_dec'             : 500,
#     'run_number'        : 5,
#     'print_iter'        : 1000,
#     'save_iter'         : 5000,
#     'batch_train_size'  : 100,
#     'num_iterations'    : 150001,
# }
# try_model(updates)

# New model; no normalization
# updates = {
#     'task'              : 'bw_to_bw_simple',
#     'save_dir'          : './savedir/bw_to_bw_simple/',
#     'normalize'         : False,
#     'new_model'         : True,
#     'use_literature_code' : False,
#     'img_size'          : 128,
#     'n_enc'             : 256,
#     'n_link'            : 100,
#     'n_latent'          : 32,
#     'n_dec'             : 256,
#     'run_number'        : 8,
#     'print_iter'        : 1000,
#     'save_iter'         : 5000,
#     'batch_train_size'  : 100,
#     'num_iterations'    : 150001,
# }
# try_model(updates)

# Not normalized; Slightly smaller encoder, bigger latent space
# updates = {
#     'task'              : 'bw_to_bw_simple',
#     'save_dir'          : './savedir/bw_to_bw_simple/',
#     'normalize'         : False,
#     'new_model'         : True,
#     'use_literature_code' : False,
#     'img_size'          : 128,
#     'n_enc'             : 256,
#     'n_link'            : 175,
#     'n_latent'          : 145,
#     'n_dec'             : 256,
#     'run_number'        : 6,
#     'print_iter'        : 1000,
#     'save_iter'         : 5000,
#     'batch_train_size'  : 100,
#     'num_iterations'    : 150001,
# }
# try_model(updates)

# updates = {
#     'task'              : 'bw_to_bw_simple',
#     'save_dir'          : './savedir/bw_to_bw_simple/',
#     'normalize'         : False,
#     'new_model'         : False,
#     'use_literature_code' : False,
#     'img_size'          : 128,
#     'n_enc'             : 256,
#     'n_link'            : 175,
#     'n_latent'          : 145,
#     'n_dec'             : 256,
#     'run_number'        : 8,
#     'print_iter'        : 1000,
#     'save_iter'         : 5000,
#     'batch_train_size'  : 100,
#     'num_iterations'    : 150001,
# }
# try_model(updates)

# 3 LAYERS debugged...
# updates = {
#     'task'              : 'bw_to_bw_simple',
#     'save_dir'          : './savedir/bw_to_bw_simple/',
#     'normalize'         : False,
#     'new_model'         : False,
#     'num_layers'        : 3,
#     'use_literature_code' : False,
#     'img_size'          : 128,
#     'n_enc'             : 500,
#     'n_latent'          : 145,
#     'n_dec'             : 500,
#     'run_number'        : 9,
#     'print_iter'        : 1000,
#     'save_iter'         : 5000,
#     'batch_train_size'  : 100,
#     'num_iterations'    : 150001,
# }
# try_model(updates)

# 3 LAYERS debugged...
# updates = {
#     'task'              : 'bw_to_bw_simple',
#     'save_dir'          : './savedir/bw_to_bw_simple/',
#     'normalize'         : False,
#     'new_model'         : False,
#     'num_layers'        : 3,
#     'use_literature_code' : False,
#     'img_size'          : 128,
#     'n_enc'             : 750,
#     'n_latent'          : 200,
#     'n_dec'             : 750,
#     'run_number'        : 9,
#     'print_iter'        : 1000,
#     'save_iter'         : 5000,
#     'batch_train_size'  : 100,
#     'num_iterations'    : 150001,
# }
# try_model(updates)

# Shitty normalizing version with 128 x 128 on literature model
# More neurons (200, instead of 138)
updates = {
    'task'              : 'bw_to_bw_simple',
    'save_dir'          : './savedir/bw_to_bw_simple/',
    'num_layers'        : 3,
    'img_size'          : 128,
    'n_enc'             : 200, # try 200 next
    'n_latent'          : 138,
    'n_dec'             : 200,
    'learning_rate'     : 0.1,
    'run_number'        : 10,
    'print_iter'        : 1000,
    'save_iter'         : 10000,
    'batch_train_size'  : 100,
    'num_iterations'    : 600001,
}
try_model(updates)

# Been using 138 and 200 neurons for 2 layer

