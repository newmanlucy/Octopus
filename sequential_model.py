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
plt.switch_backend('agg')
from parameters import *
from evo_utils import *

# Ignore tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Model:

    def __init__(self, input_data, target_data):
        # Load input and target data
        self.input_data = input_data
        self.target_data = target_data

        # Run model
        self.run_model()

        # Optimize
        self.optimize()

 
    def run_model(self):

        x = tf.reshape(self.input_data, shape=[par['batch_train_size'],*par['inp_img_shape'],1])

        conv1    = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
        conv2    = tf.layers.conv2d(inputs=maxpool1, filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
        conv3    = tf.layers.conv2d(inputs=maxpool2, filters=256, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')

        latent = tf.image.resize_images(maxpool3, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv4 = tf.layers.conv2d(inputs=latent, filters=256, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        upsample2 = tf.image.resize_images(conv4, size=(64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv5 = tf.layers.conv2d(inputs=upsample2, filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        upsample3 = tf.image.resize_images(conv5, size=(128,128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv6 = tf.layers.conv2d(inputs=upsample3, filters=par['num_conv1_filters'], kernel_size=(3,3), padding='same', activation=tf.nn.relu)

        self.latent = conv6 # (32, 32*32*256)

        logits = tf.layers.conv2d(inputs=conv6, filters=3, kernel_size=(3,3), padding='same', activation=None)
        self.output = tf.nn.relu(tf.reshape(logits, [par['batch_train_size'],par['n_output']]))
        self.o = tf.multiply(self.output, 1, name='o')
 
    def optimize(self):
        # Calculae loss
        self.loss = tf.losses.mean_squared_error(self.target_data, self.output)
        self.l = tf.multiply(self.loss, 1, name='l')
        self.train_op = tf.train.AdamOptimizer(par['learning_rate']).minimize(self.loss)


class EvoModel:

    def __init__(self):
        
        self.make_constants()
        self.make_variables()
 
    def make_variables(self):
        self.var_dict = {}
        # for i in range(par['num_conv1_filters']):
            # self.var_dict['conv1_filter{}'.format(i)] = cp.random.normal(size=(par['n_networks'],3,3,3)).astype(cp.float32)
        for i in range(3):
            self.var_dict['conv2_filter{}'.format(i)] = cp.random.normal(size=(par['n_networks'],3,3,par['num_conv1_filters'])).astype(cp.float32)

        # self.var_dict['conv1_bias'] = cp.random.normal(size=(par['n_networks'],par['inp_img_shape'][0],par['inp_img_shape'][1],par['num_conv1_filters'])).astype(cp.float32)
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
        self.input_data = cp.repeat(cp.expand_dims(to_gpu(input_data), axis=0), par['n_networks'], axis=0)
        self.target_data = to_gpu(target_data)

    def run_models(self):
        conv1 = relu(convolve(self.input_data, self.var_dict, 'conv2_filter') + cp.expand_dims(self.var_dict['conv2_bias'],axis=1))
        self.output = cp.reshape(conv1, (par['n_networks'],par['batch_train_size'],*par['out_img_shape']))

    def judge_models(self):
        img_len = par['img_size']
        self.target_data = cp.reshape(self.target_data, (par['batch_train_size'],*par['out_img_shape']))
        trimmed_img = cp.repeat(cp.expand_dims(self.target_data,axis=0),par['n_networks'],axis=0)[:,:,1:img_len-1,1:img_len-1,:]
        self.loss = cp.mean(cp.square(trimmed_img - self.output[:,:,1:img_len-1,1:img_len-1,:]),axis=(1,2,3,4))
        self.output = cp.reshape(self.output, (par['n_networks'],par['batch_train_size'],par['n_output']))

        # Rank the networks (returns [n_networks] indices)
        loss = self.loss.astype(cp.float64)
        replace_parents = cp.zeros(par['num_survivors'])
        for i in range(par['num_survivors']):
            replace_parents[i] = cp.argmin(loss[cp.range(i,par['n_networks'],par['num_survivors'])])
        salvage_migrator = par['n_networks'] - par['num_migrators'] + cp.argmin(self.loss[-par['num_migrators']:].astype(cp.float64)).astype(cp.int16)
        
        self.rank = cp.argsort(self.loss.astype(cp.float64)).astype(cp.int16)
        self.rank[:par['num_survivors']] = replace_parents
        
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
        self.con_dict['mutation_strength'] = min(0.5, self.con_dict['mutation_strength'] * 1.125)

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
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Generate stimulus
    stim = Stimulus()
    evo_model = EvoModel()

    # Placeholders for the tensorflow model
    x = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_input']], name='x')
    y = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_output']], name='y')

    # Model stats
    losses = []
    testing_losses = []

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x,y)
        
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        # Train the model
        start = time.time()
        threshold = [10000, 1000, 750, 500, 300, 150, -1]
        test_loss = [1000000]
        check_stuck = False
        stuck = 0

        for i in range(par['num_iterations']):

            # Generate training set
            input_data, conv_target, evo_target = stim.generate_train_batch()
            feed_dict = {x: input_data, y: conv_target}
            _, latent, conv_loss, conv_output = sess.run([model.train_op, model.latent, model.loss, model.output], feed_dict=feed_dict)

            if conv_loss < par['train_threshold']:
                check_stuck = True
                evo_model.load_batch(latent, evo_target)
                evo_model.run_models()
                evo_model.judge_models()
                evo_model.breed_models_genetic()
                evo_model.migration()

                evo_loss = evo_model.get_losses(True)
                if evo_loss[0] < threshold[0]:
                    threshold.pop(0)
                    if threshold[0] == 10000:
                        evo_model.slowdown_mutation(reset=True)
                    else:
                        evo_model.slowdown_mutation()
                
                if evo_loss[0] < test_loss[0]:
                    stuck = 0
                else:
                    stuck += 1
                    if stuck > 10:
                        evo_model.speed_up_mutation()
                        stuck = 0
            else:
                check_stuck = False
                evo_loss = [1000000]
                evo_output = np.array([evo_target, evo_target])

            # Check current status
            if i % par['print_iter'] == 0:

                # Print current status
                print('Model {:1} | Iter: {:4} | Mut Rate: {:.2f} | Mut Strength: {:.2f} | Stuck: {:2} | Conv Loss: {:5.3f} | Evo Loss: {} | Run Time: {:5.3f}s'.format( \
                    par['run_number'], i, evo_model.con_dict['mutation_rate'], evo_model.con_dict['mutation_strength'], stuck, conv_loss, evo_loss[0:4], time.time()-start))
                if evo_loss[0] != 1000000:
                    losses.append(evo_loss[0])

                # Save one training and output img from this iteration
                if i % par['save_iter'] == 0:
                    if evo_loss[0] < test_loss[0]:
                        # Generate batch from testing set and check the output
                        input_data, conv_target, evo_target = stim.generate_test_batch()
                        feed_dict = {x: input_data, y: conv_target}
                        latent, conv_loss, conv_output = sess.run([model.latent, model.loss, model.output], feed_dict=feed_dict)

                        evo_model.load_batch(latent, evo_target)
                        evo_model.run_models()
                        evo_model.judge_models()

                        evo_output = evo_model.output
                        if conv_loss < par['train_threshold'] + 100:
                            test_loss = evo_model.get_losses(True)
                            testing_losses.append(test_loss[0])

                        plot_outputs(conv_target, conv_output, evo_target, np.array([evo_output[0][0], evo_output[1][0]]), i)

                        pickle.dump({'var_dict':evo_model.var_dict, 'losses': losses, 'test_loss': testing_losses, 'last_iter': i}, \
                            open(par['save_dir']+'run_'+str(par['run_number'])+'_model_stats.pkl', 'wb'))
                        
                        # saved_path = saver.save(sess, './evo_model')
                        # print('model saved in {}'.format(saved_path))
                    
                    # else:
                        # test_loss[0] = evo_loss[0]
                        

                # Plot loss curve
                if i > 0:
                    plt.plot(losses[1:])
                    plt.savefig(par['save_dir']+'run_'+str(par['run_number'])+'_training_curve.png')
                    plt.close()


            
        # Generate batch from testing set and check the output
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


def plot_outputs(target_data, model_output, test_target, test_output, i):

    # Results from a training sample
    original1 = target_data[0].reshape(par['out_img_shape'])
    output1 = model_output[0].reshape(par['out_img_shape'])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original1,'Conv',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
    cv2.putText(output1,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)

    # Results from a testing sample
    trim = np.array([0,-1])
    original2 = test_target[0].reshape(par['out_img_shape'])
    original2[trim,:,:] = 0
    original2[:,trim,:] = 0
    output2 = test_output[0].reshape(par['out_img_shape'])

    original3 = test_target[1].reshape(par['out_img_shape'])
    original3[trim,:,:] = 0
    original3[:,trim,:] = 0
    output3 = test_output[1].reshape(par['out_img_shape'])
    cv2.putText(original2,'Evo',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
    cv2.putText(output2,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)

    vis1 = np.concatenate((original1, output1), axis=1)
    vis2 = np.concatenate((original2, output2), axis=1)
    vis3 = np.concatenate((original3, output3), axis=1)
    vis = np.concatenate((vis1, vis2), axis=0)
    vis = np.concatenate((vis, vis3), axis=0)
    vis = copy.deepcopy(vis)
    
    cv2.imwrite(par['save_dir']+'run_'+str(par['run_number'])+'_test_'+str(i)+'.png', vis)




if __name__ == "__main__":
    main()










