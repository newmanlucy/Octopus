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
        conv6 = tf.layers.conv2d(inputs=upsample3, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

        self.latent = tf.layers.flatten(latent) # (32, 32*32*256)

        logits = tf.layers.conv2d(inputs=conv6, filters=3, kernel_size=(3,3), padding='same', activation=None)
        if par['normalize01']:
            self.output = tf.nn.sigmoid(tf.reshape(logits, [par['batch_train_size'],par['n_output']]))
        else:
            self.output = tf.nn.relu(tf.reshape(logits, [par['batch_train_size'],par['n_output']]))
            self.o = tf.multiply(self.output, 1, name='o')
 
    def optimize(self):
        # Calculae loss
        self.loss = tf.losses.mean_squared_error(self.target_data, self.output)
        self.l = tf.multiply(self.loss, 1, name='l')
        self.train_op = tf.train.AdamOptimizer(par['learning_rate']).minimize(self.loss)

class EvoModel:

    def __init__(self, input_data, target_data, encoded_data):
        # Load input and target data
        self.input_data = input_data
        self.target_data = target_data
        self.encoded_data = encoded_data

        # Run model
        self.run_model()

        # Optimize
        self.optimize()

 
    def run_model(self):

        with tf.variable_scope('encoder'):
            self.W_in = tf.get_variable('W_in', initializer=par['W_in_init'], trainable=True)
            self.b_enc = tf.get_variable('b_enc', initializer=par['b_enc_init'], trainable=True)
            self.W_enc = tf.get_variable('W_enc', initializer=par['W_enc_init'], trainable=True)
            self.b_latent = tf.get_variable('b_latent', initializer=par['b_latent_init'], trainable=True)

        with tf.variable_scope('decoder'):
            self.W_dec = tf.get_variable('W_dec', initializer=par['W_dec_init'], trainable=True)
            self.b_dec = tf.get_variable('b_dec', initializer=par['b_dec_init'], trainable=True)
            self.W_out = tf.get_variable('W_out', initializer=par['W_out_init'], trainable=True)
            self.b_out = tf.get_variable('b_out', initializer=par['b_out_init'], trainable=True)

        # print(self.input_data.shape)
        # print(self.encoded_data.shape)
        all_input = tf.concat([self.input_data, self.encoded_data], axis=1)
        self.enc = tf.nn.sigmoid(tf.add(tf.matmul(all_input, self.W_in), self.b_enc))
        self.latent = tf.nn.sigmoid(tf.add(tf.matmul(self.enc, self.W_enc), self.b_latent))
        self.dec = tf.nn.sigmoid(tf.add(tf.matmul(self.latent, self.W_dec), self.b_dec))
        self.output = tf.nn.relu(tf.add(tf.matmul(self.dec, self.W_out), self.b_out))
 
    def optimize(self):
        # Calculae loss
        self.loss = tf.multiply(tf.losses.mean_squared_error(self.target_data, self.output), 1, name='l')
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
    evo_x = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_output']], name='evo_x')
    evo_y = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_output']], name='evo_y')
    evo_latent = tf.placeholder(tf.float32, shape=[par['batch_train_size'],32*32*256], name='evo_latent')

    # Model stats
    losses = []
    testing_losses = []

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x,y)
            evo_model = EvoModel(evo_x,evo_y,evo_latent)
        
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        # Train the model
        start = time.time()
        for i in range(par['num_iterations']):

            # Generate training set
            input_data, conv_target, evo_target = stim.generate_train_batch()
            feed_dict = {x: input_data, y: conv_target}
            _, latent, conv_loss, conv_output = sess.run([model.train_op, model.latent, model.loss, model.output], feed_dict=feed_dict)

            feed_dict = {evo_x: conv_output, evo_y: evo_target, evo_latent: latent.astype(np.float32)}
            # if conv_loss < 500:
            _, evo_loss, model_output = sess.run([evo_model.train_op, evo_model.loss, evo_model.output], feed_dict=feed_dict)
            # else:
                # evo_loss = 0
                # model_output = conv_output

            # Check current status
            if i % par['print_iter'] == 0:

                # Print current status
                print('Model {:2} | Task: {:s} | Iter: {:6} | Conv Loss: {:8.3f} | FF Loss: {:8.3f} | Run Time: {:5.3f}s'.format( \
                    par['run_number'], par['task'], i, conv_loss, evo_loss, time.time()-start))
                losses.append(evo_loss)

                # Save one training and output img from this iteration
                if i % par['save_iter'] == 0:

                    # Generate batch from testing set and check the output
                    test_input, test_target, test_target2 = stim.generate_test_batch()
                    feed_dict = {x: test_input, y: conv_target}
                    test_latent, test_loss, conv_output = sess.run([model.latent, model.loss, model.output], feed_dict=feed_dict)

                    feed_dict = {evo_x: conv_output, evo_y: evo_target, evo_latent: test_latent.astype(np.float32)}
                    test_loss, evo_output = sess.run([evo_model.loss, evo_model.output], feed_dict=feed_dict)
                    testing_losses.append(test_loss)

                    plot_outputs(test_target, conv_output, test_target2, evo_output, i)

                    # pickle.dump({'losses': losses, 'test_loss': testing_losses, 'last_iter': i}, \
                        # open(par['save_dir']+'run_'+str(par['run_number'])+'_model_stats.pkl', 'wb'))
                    
                    # saved_path = saver.save(sess, './evo_model')
                    # print('model saved in {}'.format(saved_path))

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










