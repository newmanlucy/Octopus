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
                          indent=4, sort_keys=False,
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
    'a_note'            : 'evo testing one img with changing mutation rate',
    'input_dir'         : './bw_im2/',
    'target_dir'        : './raw_im2/',
    'learning_rate'     : 0.001,
    'normalize01'       : False,
    'run_number'        : 0,
    'save_iter'         : 10,
    'print_iter'        : 1,
    'batch_train_size'  : 2,
    'num_conv1_filters' : 16,
    'n_networks'        : 50,
    'survival_rate'     : 0.1,
    'mutation_rate'     : 0.2,
    'mutation_strength' : 0.25,
    'task'              : 'conv_task'
}
try_model(updates)


