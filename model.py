"""
Catherine (Chaihyun) Lee & Lucy Newman (2018)
Contributions from Rick Stevens
"""

import tensorflow as tf
import cv2
import pickle
import time
import numpy as np
np.set_printoptions(precision=3)
import copy
from stimulus import Stimulus
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
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
        self.var_dict = {}
        for i in range(par['num_conv1_filters']):
            self.var_dict['conv1_filter{}'.format(i)] = cp.random.normal(size=(par['n_networks'],3,3,3)).astype(cp.float32)
        for i in range(3):
            self.var_dict['conv2_filter{}'.format(i)] = cp.random.normal(size=(par['n_networks'],3,3,par['num_conv1_filters'])).astype(cp.float32)

        self.var_dict['conv1_bias'] = cp.random.normal(size=(par['n_networks'],par['inp_img_shape'][0],par['inp_img_shape'][1],par['num_conv1_filters'])).astype(cp.float32)
        self.var_dict['conv2_bias'] = cp.random.normal(size=(par['n_networks'],*par['inp_img_shape'],3)).astype(cp.float32)
        self.var_dict['b_out'] = cp.random.normal(size=(par['n_networks'],par['n_output'])).astype(cp.float32)


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

        x = cp.reshape(cp.repeat(cp.expand_dims(self.input_data,axis=0),par['n_networks'],axis=0), (par['n_networks'],par['batch_train_size'],*par['inp_img_shape'],3))
        conv1 = relu(convolve(x, self.var_dict, 'conv1_filter') + cp.expand_dims(self.var_dict['conv1_bias'],axis=1))
        conv2 = relu(convolve(conv1, self.var_dict, 'conv2_filter') + cp.expand_dims(self.var_dict['conv2_bias'],axis=1))
        self.output = cp.reshape(conv2, (par['n_networks'],par['batch_train_size'],par['n_output']))
        # self.output = relu(cp.reshape(conv2, (par['n_networks'],par['batch_train_size'],par['n_output'])) + cp.expand_dims(self.var_dict['b_out'],axis=1))
        # x:      (net, 32, 128, 128, 3)
        # conv1:  (net, 32, 128, 128, 64)
        # conv2:  (net, 32, 128, 128, 3)
        # output: (net, 32, 49152)
 
    def judge_models(self):
        self.loss = cp.mean(cp.square(cp.repeat(cp.expand_dims(self.target_data,axis=0),par['n_networks'],axis=0) - self.output),axis=(1,2))

        # Rank the networks (returns [n_networks] indices)
        self.rank = cp.argsort(self.loss.astype(cp.float64)).astype(cp.int16)
        salvage_migrator = par['n_networks'] - par['num_migrators'] + cp.argmin(self.loss[-par['num_migrators']:].astype(cp.float64)).astype(cp.int16)
        if salvage_migrator not in self.rank[:par['num_survivors']]:
            self.rank[par['num_survivors']-1] = salvage_migrator

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

    def slowdown_mutation(self, reset=False):
        if reset:
            self.con_dict['mutation_rate'] = min(0.6, self.con_dict['mutation_rate'])
            self.con_dict['mutation_strength'] = min(0.45, self.con_dict['mutation_strength'])
        else:
            self.con_dict['mutation_rate'] *= 0.75
            self.con_dict['mutation_strength'] *= 0.875

    def speed_up_mutation(self):
        self.con_dict['mutation_rate'] = min(1, self.con_dict['mutation_rate']*1.25)
        self.con_dict['mutation_strength'] *= 1.125

    def breed_models_genetic(self):

        for s, name in itertools.product(range(par['num_survivors']), self.var_dict.keys()):
            indices = cp.arange(s+par['num_survivors'], par['n_networks'], par['num_survivors'])
            self.var_dict[name][indices,...] = mutate(self.var_dict[name][s,...], indices.shape[0],\
                self.con_dict['mutation_rate'], self.con_dict['mutation_strength'])

    def migration(self):
        for key in self.var_dict.keys():
            shape = self.var_dict[key][-par['num_migrators']:,...].shape
            self.var_dict[key][-par['num_migrators']:,...] = cp.random.normal(size=shape).astype(cp.float32)

def main(gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        gpu_id = None

    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Generate stimulus
    stim = Stimulus()
    evo_model = EvoModel()

    # Model stats
    losses = []
    testing_losses = []

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        # device = '/cpu:0' if gpu_id is None else '/gpu:0'
        # with tf.device(device):
            # conv_model = tf.train.import_meta_graph('conv_model_for_evo.meta', clear_devices=True)
            # conv_model.restore(sess, tf.train.latest_checkpoint('./')) 

        threshold = [10000, 1000, 750, 500, 300, 150, -1]
        test_loss = [1000000]
        stuck = 0

        # Train the model
        start = time.time()
        for i in range(par['num_iterations']):

            # Generate training set
            input_data, conv_target, evo_target = stim.generate_train_batch()
            conv_output = np.array(conv_target)
            conv_loss = 0
            # feed_dict = {'x:0': input_data, 'y:0': conv_target}
            # conv_loss, conv_output = sess.run(['l:0', 'o:0'], feed_dict=feed_dict)

            # "TRAIN" EVO MODEL
            # if conv_loss < 500:
            # else: 
            evo_model.load_batch(conv_output, evo_target)
            evo_model.run_models()
            evo_model.judge_models()
            evo_model.breed_models_genetic()
            evo_model.migration()

            evo_loss = evo_model.get_losses(True)
            if evo_loss[0] < threshold[0]:
                threshold.pop(0)
                evo_loss.slowdown_mutation()

            if evo_loss[0] < test_loss[0]:
                stuck = 0
            else:
                stuck += 1
                if stuck > 10:
                    evo_model.speed_up_mutation()
                    stuck = 0

            # Check current status
            if i % par['print_iter'] == 0:

                # Print current status
                print('Model {:1} | Iter: {:4} | Mut Rate: {:.2f} | Mut Strength: {:.2f} | Stuck: {:2} | Conv Loss: {} | Evo Loss: {} | Run Time: {:5.3f}s'.format( \
                    par['run_number'], i, evo_model.con_dict['mutation_rate'], evo_model.con_dict['mutation_strength'], stuck, conv_loss, np.array([*evo_loss[0:3],evo_loss[par['num_survivors']-1]]), time.time()-start))
                losses.append(evo_loss)

                # Save one training and output img from this iteration
                if i % par['save_iter'] == 0:
                    if evo_loss[0] < test_loss[0]:

                        # Generate batch from testing set and check the output
                        input_data, conv_target, evo_target = stim.generate_test_batch()
                        conv_output = np.array(conv_target)
                        conv_loss = 0
                        # feed_dict = {'x:0': test_input, 'y:0': conv_target}
                        # test_loss, conv_output = sess.run(['l:0', 'o:0'], feed_dict=feed_dict)

                        # "TEST" EVO MODEL
                        evo_model.load_batch(conv_output, evo_target)
                        evo_model.run_models()
                        evo_model.judge_models()

                        evo_output = evo_model.output
                        test_loss = evo_model.get_losses(True)
                        testing_losses.append(test_loss)

                        plot_outputs(conv_target, conv_output, evo_target, np.array([evo_output[0][0],evo_output[1][0]]), i)

                        pickle.dump({'var_dict':evo_model.var_dict, 'losses': losses, 'test_loss': testing_losses, 'last_iter': i}, \
                            open(par['save_dir']+'run_'+str(par['run_number'])+'_model_stats.pkl', 'wb'))
                    
                    # FIGURE OUT HOW TO SAVE EVO MODEL

                # Plot loss curve
                if i > 0:
                    plt.plot([l[0]for l in losses[1:]])
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
    original2 = test_target[0].reshape(par['out_img_shape'])
    output2 = test_output[0].reshape(par['out_img_shape'])
    original3 = test_target[1].reshape(par['out_img_shape'])
    output3 = test_output[1].reshape(par['out_img_shape'])
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



if __name__ == "__main__":
    main()











