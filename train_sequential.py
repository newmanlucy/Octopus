import json
import time
import sequential_model as model
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
    'a_note'            : 'latent input, one image, later training + trim border',
    'input_dir'         : './bw_im2/',
    'target_dir'        : './raw_im2/',
    'batch_train_size'  : 2,
    'run_number'        : 9,
    'save_iter'         : 5,
    'print_iter'        : 1,
    'num_conv1_filters' : 64,
    'n_networks'        : 75,
    'survival_rate'     : 0.12,
    'mutation_rate'     : 0.6,
    'mutation_strength' : 0.45,
    'train_threshold'   : 150,
    'task'              : 'conv_task'
}
try_model(updates)


