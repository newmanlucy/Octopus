"""
Catherine (Chaihyun) Lee & Lucy Newman (2018)
Contributions from Rick Stevens
"""

import tensorflow as tf
import cv2
import pickle
import time
import numpy as np
import copy
from stimulus import Stimulus
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from parameters import *
from evo_utils import *

# Ignore tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


"""
The plan right now is to load up a convolutional model that is trained 
to solve bw1 to bw3 and then use the output of the convolutional model
to train EvoModel through evolutionary algorithm
"""
class EvoModel:

    def __init__(self):
        
        self.make_constants()
        self.make_variables()
 
    def make_variables(self):
        var_names = ['conv1_bias','conv2_bias','b_out']
        for i in range(64):
            var_names.append('conv1_filter{}'.format(i))
        for i in range(3):
            var_names.append('conv2_filter{}'.format(i))


        self.var_dict = {}
        for v in var_names:
            self.var_dict[v] = to_gpu(par[v+'_init'])

    def make_constants(self):
        constants = ['n_networks','mutation_rate','mutation_strength','cross_rate']

        self.con_dict = {}
        for c in constants:
            self.con_dict[c] = to_gpu(par[c])

    def update_constant(self, name, val):
        self.con_dict[name] = to_gpu(val)

    def load_batch(self, input_data, target_data):
        self.input_data = to_gpu(input_data)
        self.target_data = to_gpu(target_data)

    def run_models(self):

        # Add n_networks dimension
        # x = cp.reshape(self.input_data, shape=[par['batch_train_size'],*par['inp_img_shape'],3])
        # conv1  = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        # conv2 = tf.layers.conv2d(inputs=conv1, filters=3, kernel_size=(3,3), padding='same', activation=None)
        # self.output = tf.nn.relu(tf.reshape(conv2, [par['batch_train_size'],par['n_output']]))
        
        x = cp.reshape(self.input_data, (par['batch_train_size'],*par['inp_img_shape'],3))
        conv1 = cp.zeros((par['n_networks'], par['batch_train_size'],par['inp_img_shape'][0],par['inp_img_shape'][1],64))
        conv2 = cp.zeros((par['n_networks'], par['batch_train_size'],*par['inp_img_shape'],3))
        self.output = relu(np.reshape(conv2, (par['batch_train_size'],par['n_output']))#cp.zeros((par['n_networks'], par['batch_train_size'],par['n_output']))
        # x:      (net, 32, 128, 128, 3)
        # conv1:  (net, 32, 128, 128, 64)
        # conv2:  (net, 32, 128, 128, 3)
        # output: (net, 32, 49152)
 
    def judge_models(self):
        self.loss = cp.mean(cp.square(self.target_data - self.output))

        # Rank the networks (returns [n_networks] indices)
        self.rank = cp.argsort(self.loss.astype(cp.float32)).astype(cp.int16)
        for name in self.var_dict.keys():
            self.var_dict[name] = self.var_dict[name][self.rank,...]

    def get_weights(self):
        return to_cpu({name:cp.mean(self.var_dict[name][:par['num_survivors'],...], axis=0) \
            for name in self.var_dict.keys()})

    def get_losses(self, ranked=True):
        if ranked:
            return to_cpu(self.loss[self.rank])
        else:
            return to_cpu(self.loss)

    def breed_models_genetic(self):
        for s, name in itertools.product(range(par['num_survivors']), self.var_dict.keys()):
            indices = cp.arange(s+par['num_survivors'], par['n_networks'], par['num_survivors'])

            self.var_dict[name][indices,...] = mutate(self.var_dict[name][s,...], indices.shape[0],\
                self.con_dict['mutation_rate'], self.con_dict['mutation_strength'])


def main(gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Generate stimulus
    stim = Stimulus()
    evo_model = EvoModel()

    # Model stats
    losses = []
    testing_losses = []

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            conv_model = tf.train.import_meta_graph('conv_model_for_evo.meta', clear_devices=True)
            conv_model.restore(sess, tf.train.latest_checkpoint('./')) 


        # Train the model
        start = time.time()
        for i in range(par['num_iterations']):

            # Generate training set
            input_data, conv_target, evo_target = stim.generate_train_batch()
            feed_dict = {'x:0': input_data, 'y:0': conv_target}
            conv_loss, conv_output = sess.run(['l:0', 'o:0'], feed_dict=feed_dict)

            # "TRAIN" EVO MODEL
            # if conv_loss < 500:
            # else: 
            evo_model.load_batch(conv_output, evo_target)
            evo_model.run_models()
            evo_model.judge_models()
            
            evo_loss = evo_model.get_losses(True)
            evo_model.breed_models_genetic()

            # Check current status
            if i % par['print_iter'] == 0:

                # Print current status
                print('Model {:2} | Task: {:s} | Iter: {:6} | Conv Loss: {:8.3f} | Evo Loss: {:8.3f} | Run Time: {:5.3f}s'.format( \
                    par['run_number'], par['task'], i, conv_loss, evo_loss[0], time.time()-start))
                losses.append(evo_loss)

                # Save one training and output img from this iteration
                if i % par['save_iter'] == 0:

                    # Generate batch from testing set and check the output
                    test_input, test_target, test_target2 = stim.generate_test_batch()
                    feed_dict = {'x:0': test_input, 'y:0': conv_target}
                    test_loss, conv_output = sess.run(['l:0', 'o:0'], feed_dict=feed_dict)

                    # "TEST" EVO MODEL
                    evo_model.load_batch(conv_output, test_target2)
                    evo_model.run_models()
                    evo_model.judge_models()

                    evo_output = evo_model.output
                    test_loss = evo_model.get_losses(True)
                    testing_losses.append(test_loss)

                    plot_outputs(test_target, conv_output, test_target2, evo_output[0], i)

                    pickle.dump({'losses': losses, 'test_loss': testing_losses, 'last_iter': i}, \
                        open(par['save_dir']+'run_'+str(par['run_number'])+'_model_stats.pkl', 'wb'))
                    
                    # FIGURE OUT HOW TO SAVE EVO MODEL

                # Plot loss curve
                if i > 0:
                    plt.plot(losses[1:])
                    plt.savefig(par['save_dir']+'run_'+str(par['run_number'])+'_training_curve.png')
                    plt.close()


            
        # Test the model
        # test_input, test_target = stim.generate_test_batch()
        # feed_dict = {x: test_input, y: test_target}
        # test_loss, test_output = sess.run([model.loss, model.output], feed_dict=feed_dict)
        # print("FINAL TEST LOSS IS: ", test_loss)

        # plt.plot(testing_losses)
        # plt.savefig(par['save_dir']+'run_test_'+str(par['run_number'])+'_testing_curve.png')
        # plt.close()

        # for i in range(10):
        #     idx = [i, i+10, i+20]
        #     plot_testing(test_target[idx], test_output[idx], i)




def plot_testing(test_target, test_output, i):

    # Results from a testing sample
    original1 = test_target[0].reshape(par['out_img_shape'])
    output1 = test_output[0].reshape(par['out_img_shape'])
    original2 = test_target[1].reshape(par['out_img_shape'])
    output2 = test_output[1].reshape(par['out_img_shape'])
    original3 = test_target[2].reshape(par['out_img_shape'])
    output3 = test_output[2].reshape(par['out_img_shape'])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original1,'Testing',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
    cv2.putText(output1,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)

    vis1 = np.concatenate((original1, output1), axis=1)
    vis2 = np.concatenate((original2, output2), axis=1)
    vis3 = np.concatenate((original3, output3), axis=1)
    vis = np.concatenate((vis1, vis2), axis=0)
    vis = np.concatenate((vis, vis3), axis=0)
    vis = copy.deepcopy(vis)
    if par['normalize01']:
        print("UN-NORMALIZE")
        if np.max(vis) > 1 or np.min(vis) < 0:
            print(np.max(vis))
            print(np.min(vis))
            print("Something is wrong")
            quit()
        vis *= 255
        vis = np.int16(vis)

    cv2.imwrite(par['save_dir']+'run_test_'+str(par['run_number'])+'_output_'+str(i)+'.png', vis)


def plot_outputs(target_data, model_output, test_target, test_output, i):

    # Results from a training sample
    original1 = target_data[0].reshape(par['out_img_shape'])
    output1 = model_output[0].reshape(par['out_img_shape'])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original1,'Conv',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
    cv2.putText(output1,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)

    # Results from a testing sample
    original2 = test_target[1].reshape(par['out_img_shape'])
    output2 = test_output[1].reshape(par['out_img_shape'])
    original3 = test_target[2].reshape(par['out_img_shape'])
    output3 = test_output[2].reshape(par['out_img_shape'])
    cv2.putText(original2,'Evo',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
    cv2.putText(output2,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)

    vis1 = np.concatenate((original1, output1), axis=1)
    vis2 = np.concatenate((original2, output2), axis=1)
    vis3 = np.concatenate((original3, output3), axis=1)
    vis = np.concatenate((vis1, vis2), axis=0)
    vis = np.concatenate((vis, vis3), axis=0)
    vis = copy.deepcopy(vis)
    if par['normalize01']:
        print("UN-NORMALIZE")
        if np.max(vis) > 1 or np.min(vis) < 0:
            print(np.max(vis))
            print(np.min(vis))
            print("Something is wrong")
            quit()
        vis *= 255
        vis = np.int16(vis)

    cv2.imwrite(par['save_dir']+'run_'+str(par['run_number'])+'_test_'+str(i)+'.png', vis)


def eval_weights():

    """ NEED TO FIX """
    with tf.variable_scope('encoder', reuse=True):
        W_in = tf.get_variable('W_in')
        b_enc = tf.get_variable('b_enc')

    with tf.variable_scope('decoder', reuse=True):
        W_dec = tf.get_variable('W_dec')
        b_dec = tf.get_variable('b_dec')
        W_out = tf.get_variable('W_out')
        b_out = tf.get_variable('b_out')

    weights = {
        'W_in'  : W_in.eval(),
        'W_dec' : W_dec.eval(),
        'W_out' : W_out.eval(),
        'b_enc' : b_enc.eval(),
        'b_dec' : b_dec.eval(),
        'b_out' : b_out.eval(),
    }

    if par['num_layers'] >= 3:
        with tf.variable_scope('encoder', reuse=True):
            W_enc = tf.get_variable('W_enc')
            b_latent = tf.get_variable('b_latent')

        weights['W_enc'] = W_enc.eval()
        weights['b_latent'] = b_latent.eval()

    if par['num_layers'] == 5:
        with tf.variable_scope('encoder', reuse=True):
            W_link = tf.get_variable('W_link')
            b_link = tf.get_variable('b_link')

        with tf.variable_scope('decoder', reuse=True):
            W_link2 = tf.get_variable('W_link2')
            b_link2 = tf.get_variable('b_link2')

        weights['W_link'] = W_link.eval()
        weights['b_link'] = b_link.eval()
        weights['W_link2'] = W_link2.eval()
        weights['b_link2'] = b_link2.eval()

    return weights


if __name__ == "__main__":
    main()











