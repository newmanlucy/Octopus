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

# Ignore tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

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

        self.latent = tf.multiply(conv6, 1, name='encoded')
        logits = tf.layers.conv2d(inputs=conv6, filters=3, kernel_size=(3,3), padding='same', activation=None)
        self.output = tf.multiply(tf.nn.relu(tf.reshape(logits, [par['batch_train_size'],par['n_output']])), 1, name='o')
 
    def optimize(self):
        # Calculae loss
        self.loss = tf.multiply(tf.losses.mean_squared_error(self.target_data, self.output), 1, name='l')
        self.train_op = tf.train.AdamOptimizer(par['learning_rate']).minimize(self.loss)


class EvoModel:

    def __init__(self, input_data, target_data):
        # Load input and target data
        self.input_data = input_data
        self.target_data = target_data

        # Run model
        self.run_model()

        # Optimize
        self.optimize()

 
    def run_model(self):

        x = tf.reshape(self.input_data, shape=[par['batch_train_size'],*par['inp_img_shape'],3])

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
        conv6 = tf.layers.conv2d(inputs=upsample3, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

        logits = tf.layers.conv2d(inputs=conv6, filters=3, kernel_size=(3,3), padding='same', activation=None)
        self.output = tf.nn.relu(tf.reshape(logits, [par['batch_train_size'],par['n_output']]))
        self.o = tf.multiply(self.output, 1, name='o')
 
    def optimize(self):
        # Calculae loss
        self.loss = tf.losses.mean_squared_error(self.target_data, self.output)
        self.l = tf.multiply(self.loss, 1, name='l')
        self.train_op = tf.train.AdamOptimizer(par['learning_rate']).minimize(self.loss)


def main(gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Generate stimulus
    stim = Stimulus()

    # Placeholders for the tensorflow model
    x = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_input']], name='x')
    y = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_output']], name='y')

    # Model stats
    losses = []
    testing_losses = []

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x,y)
        
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        # Train the model
        start = time.time()
        for i in range(par['num_iterations']):

            # Generate training set
            input_data, target_data, _ = stim.generate_train_batch()
            feed_dict = {x: input_data, y: target_data}
            _, train_loss, model_output = sess.run([model.train_op, model.loss, model.output], feed_dict=feed_dict)

            # Check current status
            if i % par['print_iter'] == 0:

                # Print current status
                print('Model {:2} | Task: {:s} | Iter: {:6} | Loss: {:8.3f} | Run Time: {:5.3f}s'.format( \
                    par['run_number'], par['task'], i, train_loss, time.time()-start))
                losses.append(train_loss)

                # Save one training and output img from this iteration
                if i % par['save_iter'] == 0:

                    # Generate batch from testing set and check the output
                    test_input, test_target, _ = stim.generate_test_batch()
                    feed_dict = {x: test_input, y: test_target}
                    test_loss, test_output = sess.run([model.loss, model.output,], feed_dict=feed_dict)
                    testing_losses.append(test_loss)

                    plot_outputs(target_data, model_output, test_target, test_output, i)

                    pickle.dump({'losses': losses, 'test_loss': testing_losses, 'last_iter': i}, \
                        open(par['save_dir']+'run_'+str(par['run_number'])+'_model_stats.pkl', 'wb'))
                    
                    saved_path = saver.save(sess, './conv_model_with_latent')
                    print('model saved in {}'.format(saved_path))

                # Plot loss curve
                if i > 0:
                    plt.plot(losses[1:])
                    plt.savefig(par['save_dir']+'run_'+str(par['run_number'])+'_training_curve.png')
                    plt.close()

def plot_outputs(target_data, model_output, test_target, test_output, i):

    # Results from a training sample
    original1 = target_data[0].reshape(par['out_img_shape'])
    output1 = model_output[0].reshape(par['out_img_shape'])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original1,'Conv',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
    cv2.putText(output1,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)

    # Results from a testing sample
    if par['batch_train_size'] == 1:
        original2 = test_target[0].reshape(par['out_img_shape'])
        output2 = test_output[0].reshape(par['out_img_shape'])
        original3 = test_target[0].reshape(par['out_img_shape'])
        output3 = test_output[0].reshape(par['out_img_shape'])
    else:
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
            'a_note'            : 'conv_model with latent, batch1, filt 16',
            'input_dir'         : './bw_im/',
            'target_dir'        : './raw_im/',
            'batch_train_size'  : 1,
            'learning_rate'     : 0.001,
            'normalize01'       : False,
            'num_conv1_filters' : 16,
            'run_number'        : 11,
            "save_iter"         : 100,
            'task'              : 'conv_task',
            'one_img'           : False,
            'simulation'        : False
        }
        update_parameters(updates)
        print('Model number ' + str(par['run_number']) + ' running!')
        main(gpu_id)
        print('Model run concluded.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))










