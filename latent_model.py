"""
Catherine (Chaihyun) Lee & Lucy Newman (2018)
Contributions from Rick Stevens
"""

import tensorflow as tf
import io
import json
import cv2
import pickle
import time
from collections import OrderedDict 
import numpy as np
np.set_printoptions(precision=3)
import copy
from stimulus import Stimulus
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from parameters import *
from evo_utils import *
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

# Ignore tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


"""
The plan right now is to load up a convolutional model that is trained 
to solve bw1 to bw3 and then use the output of the convolutional model
to train EvoModel through evolutionary algorithm
"""
class EvoModel:

    def __init__(self):
        
        self.make_constants()
        self.make_variables()
        self.original = True
        self.iter = 0
 
    def make_variables(self):
        self.var_dict = OrderedDict()
        for i in range(3):
            self.var_dict['conv2_filter{}'.format(i)] = cp.random.normal(size=(par['n_networks'],3,3,par['num_conv1_filters'])).astype(cp.float32)

        self.var_dict['conv2_bias'] = cp.random.normal(size=(par['n_networks'],*par['inp_img_shape'],3)).astype(cp.float32)
        self.var_dict['b_out'] = cp.random.normal(size=(par['n_networks'],par['n_output'])).astype(cp.float32)

    def update_variables(self, updates):
        for key, val in updates.items():
            self.var_dict[key] = to_gpu(val)

    def make_constants(self):
        constants = ['n_networks','mutation_rate','mutation_strength','cross_rate']

        self.con_dict = {}
        for c in constants:
            self.con_dict[c] = to_gpu(par[c])

    def update_constant(self, name, val):
        self.con_dict[name] = to_gpu(val)

    def load_batch(self, input_data, target_data):
        self.input_data = cp.repeat(cp.expand_dims(to_gpu(input_data), axis=0), par['n_networks'], axis=0)
        self.target_data = to_gpu(target_data)

    def run_models(self):
        conv1 = relu(convolve(self.input_data, self.var_dict) + cp.expand_dims(self.var_dict['conv2_bias'],axis=1))
        self.output = cp.reshape(conv1, (par['n_networks'],par['batch_train_size'],*par['out_img_shape']))

    def judge_models(self):
        img_len = par['img_size']
        self.target_data = cp.reshape(self.target_data, (par['batch_train_size'],*par['out_img_shape']))
        trimmed_img = cp.repeat(cp.expand_dims(self.target_data,axis=0),par['n_networks'],axis=0)[:,:,1:img_len-1,1:img_len-1,:]
        
        self.loss = cp.mean(cp.square(trimmed_img - self.output[:,:,1:img_len-1,1:img_len-1,:]),axis=(1,2,3,4)).astype(cp.float64)
        self.output = cp.reshape(self.output, (par['n_networks'],par['batch_train_size'],par['n_output']))
        self.rank = cp.argsort(self.loss).astype(cp.int16)

        if self.original:
            self.original = False
        else:
            loss = self.loss.astype(cp.float64)
            replace_parents = cp.zeros(par['num_survivors']).astype(cp.int16)
            for i in range(par['num_survivors']):
                replace_parents[i] = cp.argmin(loss[cp.arange(i,par['n_networks'],par['num_survivors'])])*par['num_survivors'] + i
            
            temp = copy.deepcopy(replace_parents)
            for i,idx in enumerate(cp.argsort(self.loss[replace_parents])):
                replace_parents[i] = temp[idx]

            self.rank = (cp.ones(par['n_networks'])*(par['n_networks']-1)).astype(cp.int16)
            self.rank[:par['num_survivors']] = replace_parents
            
            salvage_migrator = par['n_networks'] - par['num_migrators'] + cp.argmin(self.loss[-par['num_migrators']:].astype(cp.float64)).astype(cp.int16)
            if par['num_migrators'] > 0 and salvage_migrator not in self.rank[:par['num_survivors']]:
                self.rank[par['num_survivors']-1] = salvage_migrator

        for name in self.var_dict.keys():
            self.var_dict[name][:par['num_survivors']] = self.var_dict[name][self.rank[:par['num_survivors']],...]


    def get_weights(self):
        return to_cpu({name:cp.mean(self.var_dict[name][:par['num_survivors'],...], axis=0) \
            for name in self.var_dict.keys()})

    def get_losses(self, ranked=True):
        if ranked:
            return to_cpu(self.loss[self.rank])
        else:
            return to_cpu(self.loss)

    def slowdown_mutation(self, reset=False):
        if reset:
            self.con_dict['mutation_rate'] = min(0.6, self.con_dict['mutation_rate'])
            self.con_dict['mutation_strength'] = min(0.45, self.con_dict['mutation_strength'])
        else:
            self.con_dict['mutation_rate'] = max(0.1, self.con_dict['mutation_rate'] * 0.7)
            self.con_dict['mutation_strength'] = max(0.05, self.con_dict['mutation_strength'] * 0.8)
            # self.con_dict['mutation_rate'] *= 0.7
            # self.con_dict['mutation_strength'] *= 0.8

    def speed_up_mutation(self):
        self.con_dict['mutation_rate'] = min(0.7, self.con_dict['mutation_rate']*1.25)
        self.con_dict['mutation_strength'] = min(0.45, self.con_dict['mutation_strength'] * 1.125)

    def breed_models_genetic(self):
        for s, name in itertools.product(range(par['num_survivors']), self.var_dict.keys()):
            indices = cp.arange(s+par['num_survivors'], par['n_networks'], par['num_survivors'])
            self.var_dict[name][indices,...] = mutate(self.var_dict[name][s,...], indices.shape[0],\
                self.con_dict['mutation_rate'], self.con_dict['mutation_strength'])

    def migration(self):
        for key in self.var_dict.keys():
            shape = self.var_dict[key][-par['num_migrators']:,...].shape
            self.var_dict[key][-par['num_migrators']:,...] = cp.random.normal(size=shape).astype(cp.float32)

class ConvModel():
    def __init__(self, input_data, target_data):
        # Load input and target data
        self.input_data = input_data
        self.target_data = target_data

        # Run model
        self.run_model()

        # Optimize
        self.optimize()

 
    def run_model(self):

        self.latent = tf.multiply(self.input_data, 1, name='encoded')
        logits = tf.layers.conv2d(inputs=conv6, filters=3, kernel_size=(3,3), padding='same', activation=None)
        self.output = tf.multiply(tf.nn.relu(tf.reshape(logits, [par['batch_train_size'],par['n_output']])), 1, name='o')
 
    def optimize(self):
        # Calculae loss
        self.loss = tf.multiply(tf.losses.mean_squared_error(self.target_data, self.output), 1, name='l')
        self.train_op = tf.train.AdamOptimizer(par['learning_rate']).minimize(self.loss)

def main(gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        # gpu_id = None

    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Generate stimulus
    stim = Stimulus()
    evo_model = EvoModel()

    # print('Loading evo model!')
    # saved_evo_model = pickle.load(open('./savedir/conv_task/run_14_model_stats.pkl','rb'))
    # best_weights = {}
    # for key, val in saved_evo_model['var_dict'].items():
        # best_weights[key] = val
    # evo_model.update_variables(best_weights)

    # Model stats
    losses = []
    testing_losses = []

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            # g1 = tf.Graph()
            # with g1.as_default():
                folder = './latent_all_img_batch16_filt16_loss80/'
                conv_model = tf.train.import_meta_graph(folder + 'conv_model_with_latent.meta', clear_devices=True)
                conv_model.restore(sess, tf.train.latest_checkpoint(folder)) 
                print('Loaded model from',folder)
            
            # g2 = tf.Graph()
            # with g2.as_default():
                # x = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_input']])
                # y = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_output']])
                


        threshold = [10000, 5000, 1000, 750, 500, 300, 150, -1]
        test_loss = [1000000]
        stuck = 0

        # Train the model
        start = time.time()
        for i in range(par['num_iterations']):

            # Generate training set
            input_data, conv_target, evo_target = stim.generate_train_batch()
            feed_dict = {'x:0': input_data, 'y:0': conv_target}
            conv_loss, conv_output, encoded = sess.run(['l:0', 'o:0','encoded:0'], feed_dict=feed_dict)

            # "TRAIN" EVO MODEL
            evo_model.load_batch(encoded, evo_target)
            evo_model.run_models()
            evo_model.judge_models()
            evo_model.breed_models_genetic()
            if par['num_migrators'] > 0:
                evo_model.migration()

            evo_loss = evo_model.get_losses(True)
            if evo_loss[0] < threshold[0]:
                threshold.pop(0)
                if threshold[0] == 10000:
                    evo_model.slowdown_mutation(reset=True)
                    par['num_migrators'] = 0
                else:
                    evo_model.slowdown_mutation()

            if evo_loss[0] < test_loss[0]:
                stuck = 0
            else:
                stuck += 1
                if stuck > 20:
                    # evo_model.speed_up_mutation()
                    evo_model.slowdown_mutation()
                    stuck = 0

            # Check current status
            if i % par['print_iter'] == 0:

                # Print current status
                print('Model {:1} | Iter: {:4} | Mut Rate: {:.2f} | Mut Strength: {:.2f} | Stuck: {:2} | Conv Loss: {:.2f} | Evo Loss: {} | Run Time: {:5.3f}s'.format( \
                    par['run_number'], i, evo_model.con_dict['mutation_rate'], evo_model.con_dict['mutation_strength'], stuck, conv_loss, np.array([*evo_loss[0:3],evo_loss[par['num_survivors']-1]]), time.time()-start))
                if evo_loss[0] != 10000:
                    losses.append(evo_loss[0])

                # Save one training and output img from this iteration
                if i % par['save_iter'] == 0:
                    if evo_loss[0] < test_loss[0]:
                        print("SAVING",par['save_dir'])
                        # Generate batch from testing set and check the output
                        input_data, conv_target, evo_target = stim.generate_test_batch()
                        feed_dict = {'x:0': input_data, 'y:0': conv_target}
                        test_loss, conv_output, encoded = sess.run(['l:0', 'o:0','encoded:0'], feed_dict=feed_dict)

                        # "TEST" EVO MODEL
                        evo_model.load_batch(encoded, evo_target)
                        evo_model.run_models()
                        evo_model.judge_models()

                        evo_output = evo_model.output
                        test_loss = evo_model.get_losses(True)
                        testing_losses.append(test_loss[0])

                        plot_outputs(conv_target, conv_output, evo_target, evo_output, i)

                        pickle.dump({'var_dict':evo_model.var_dict, 'losses': losses, 'test_loss': testing_losses, 'last_iter': i}, \
                            open(par['save_dir']+'run_'+str(par['run_number'])+'_model_stats.pkl', 'wb'))
                    
                    # FIGURE OUT HOW TO SAVE EVO MODEL

                # Plot loss curve
                if i > 0:
                    plt.plot(losses[1:])
                    plt.savefig(par['save_dir']+'run_'+str(par['run_number'])+'_training_curve.png')
                    plt.close()


def plot_outputs(target_data, model_output, test_target, test_output, i):

    # Results from a training sample
    outputs = []
    for b in range(4):
        batch = b
        original1 = target_data[batch].reshape(par['out_img_shape'])
        output1 = model_output[batch].reshape(par['out_img_shape'])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(original1,'Conv',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
        cv2.putText(output1,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)

        # Results from a testing sample
        original2 = test_target[batch].reshape(par['out_img_shape'])
        output2 = test_output[0][batch].reshape(par['out_img_shape'])
        output3 = test_output[1][batch].reshape(par['out_img_shape'])
        output4 = test_output[2][batch].reshape(par['out_img_shape'])
        cv2.putText(original2,'Evo',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
        cv2.putText(output2,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
        cv2.putText(output3,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
        cv2.putText(output4,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)

        vis1 = np.concatenate((original1, output1), axis=1)
        vis2 = np.concatenate((original2, output2), axis=1)
        vis3 = np.concatenate((original2, output3), axis=1)
        vis4 = np.concatenate((original2, output4), axis=1)
        vis = np.concatenate((vis1, vis2), axis=0)
        vis = np.concatenate((vis, vis3), axis=0)
        vis = np.concatenate((vis, vis4), axis=0)
        outputs.append(vis)

    if len(outputs) == 1:
        vis = outputs[0]
    else:
        vis = np.concatenate((outputs[0],outputs[1]), axis=1)
        for batch in range(2,len(outputs)):
            vis = np.concatenate((vis,outputs[batch]), axis=1)

    cv2.imwrite(par['save_dir']+'run_'+str(par['run_number'])+'_test_'+str(i)+'.png', vis)


if __name__ == "__main__":
    # GPU
    try:
        gpu_id = sys.argv[1]
    except:
        gpu_id = None

    # Run Model
    t0 = time.time()
    try:
        updates = {
            'a_note'            : 'latent to evo, raw_im1, lower mutation rate',
            'print_iter'        : 1,
            'save_iter'         : 5,
            'batch_train_size'  : 16,
            'run_number'        : 23,
            'num_conv1_filters' : 16,
            'n_networks'        : 65,
            'survival_rate'     : 0.12,
            'mutation_rate'     : 0.6,
            'mutation_strength' : 0.45,
            'migration_rate'    : 0.1,
            'task'              : 'conv_task',
            'one_img'           : False,
            'simulation'        : False
        }
        # Save updated parameters
        update_parameters(updates)
        with io.open(par['save_dir']+'run_'+str(par['run_number'])+'_params.json', 'w', encoding='utf8') as outfile:
            str_ = json.dumps(updates,
                              indent=4, sort_keys=False,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))

        print('Model number ' + str(par['run_number']) + ' running!')
        main(gpu_id)
        print('Model run concluded.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))





















