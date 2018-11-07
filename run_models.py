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


updates = {
    'a_note'            : 'testing normalization with relu and linear at end',
    'batch_train_size'  : 20,
    'learning_rate'     : 0.001,
    'n_dec'             : 200,
    'n_enc'             : 200,
    'n_latent'          : 138,
    'normalize01'       : False,
    'num_layers'        : 3,
    'run_number'        : 14,
    "save_iter"         : 10000,
    'task'              : 'bw_to_bw_simple'
}
try_model(updates)


