import json
import time
import model
import io
from parameters import *
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

def try_model(updates):

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
        model.main()
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
updates = {
    'task'              : 'bw_to_bw_simple',
    'save_dir'          : './savedir/bw_to_bw_simple/',
    'img_size'          : 28,
    'n_enc'             : 500,
    'n_latent'          : 32,
    'n_dec'             : 500,
    'run_number'        : 1,
    'print_iter'        : 1000,
    'batch_train_size'  : 100,
    'num_iterations'    : 30001,
}
try_model(updates)


