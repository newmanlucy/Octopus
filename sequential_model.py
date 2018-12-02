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
        self.output = cp.reshape(conv1, (par['n_networks'],par['batch_train_size'],par['n_output']))

    def judge_models(self):
        self.loss = cp.mean(cp.square(cp.repeat(cp.expand_dims(self.target_data,axis=0),par['n_networks'],axis=0) - self.output),axis=(1,2))

        # Rank the networks (returns [n_networks] indices)
        self.rank = cp.argsort(self.loss.astype(cp.float64)).astype(cp.int16)
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

    def slowdown_mutation(self):
        self.con_dict['mutation_rate'] *= 0.75
        self.con_dict['mutation_strength'] *= 0.875

    def speed_up_mutation(self):
        self.con_dict['mutation_rate'] *= 1.25
        self.con_dict['mutation_strength'] *= 1.125

    def breed_models_genetic(self):
        for s, name in itertools.product(range(par['num_survivors']), self.var_dict.keys()):
            indices = cp.arange(s+par['num_survivors'], par['n_networks'], par['num_survivors'])

            self.var_dict[name][indices,...] = mutate(self.var_dict[name][s,...], indices.shape[0],\
                self.con_dict['mutation_rate'], self.con_dict['mutation_strength'])


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
        threshold = [1000, 750, 500, 300, 150, -1]
        test_loss = [1000000]
        stuck = 0

        for i in range(par['num_iterations']):

            # Generate training set
            input_data, conv_target, evo_target = stim.generate_train_batch()
            feed_dict = {x: input_data, y: conv_target}
            _, latent, conv_loss, conv_output = sess.run([model.train_op, model.latent, model.loss, model.output], feed_dict=feed_dict)

            if conv_loss < 600:
                evo_model.load_batch(latent, evo_target)
                evo_model.run_models()
                evo_model.judge_models()
                evo_model.breed_models_genetic()

                evo_loss = evo_model.get_losses(True)
                if evo_loss[0] < threshold[0]:
                    threshold.pop(0)
                    evo_loss.speed_up_mutation()
            else:
                evo_loss = test_loss
                evo_output = np.array([evo_target, evo_target])

            # Check current status
            if i % par['print_iter'] == 0:

                # Print current status
                print('Model {:2} | Task: {:s} | Iter: {:6} | Conv Loss: {:8.3f} | Evo Loss: {} | Run Time: {:5.3f}s'.format( \
                    par['run_number'], par['task'], i, conv_loss, evo_loss[0:4], time.time()-start))
                losses.append(evo_loss[0])

                # Save one training and output img from this iteration
                if i % par['save_iter'] == 0:
                    if evo_loss[0] < test_loss[0]:
                        stuck = 0

                        # Generate batch from testing set and check the output
                        input_data, conv_target, evo_target = stim.generate_test_batch()
                        feed_dict = {x: input_data, y: conv_target}
                        latent, conv_loss, conv_output = sess.run([model.latent, model.loss, model.output], feed_dict=feed_dict)

                        evo_model.load_batch(latent, evo_target)
                        evo_model.run_models()
                        evo_model.judge_models()

                        evo_output = evo_model.output
                        if conv_loss < 750:
                            test_loss = evo_model.get_losses(True)
                            testing_losses.append(test_loss[0])

                        plot_outputs(conv_target, conv_output, evo_target, np.array([evo_output[0][0], evo_output[1][0]]), i)

                        pickle.dump({'var_dict':evo_model.var_dict, 'losses': losses, 'test_loss': testing_losses, 'last_iter': i}, \
                            open(par['save_dir']+'run_'+str(par['run_number'])+'_model_stats.pkl', 'wb'))
                        
                        # saved_path = saver.save(sess, './evo_model')
                        # print('model saved in {}'.format(saved_path))
                    
                    else:
                        test_loss[0] = evo_loss[0]
                        stuck += 1
                        if stuck > 20:
                            evo_model.speed_up_mutation()

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










