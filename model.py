"""
Catherine (Chaihyun) Lee & Lucy Newman (2018)
Contributions from Rick Stevens
"""

import tensorflow as tf
import cv2
import pickle
import time
import numpy as numpy
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


    def conv2d(self, x, W):

        return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME'))


    def max_pool2d(self, x):

        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

 
    def run_model(self):

        # 5x5 convolution with 1 inputs and 32 outputs
        weights = {'W_conv1': tf.Variable(tf.random_normal([3,3,1,64])), #used to be 5,5
                   'W_conv2': tf.Variable(tf.random_normal([3,3,64,128])), 
                   'W_conv3': tf.Variable(tf.random_normal([3,3,128,256])),
                   'W_fc': tf.Variable(tf.random_normal([16*16*256,256])), 
                   'W_out': tf.Variable(tf.random_normal([256,par['n_output']]))}

        biases = {'b_conv1': tf.Variable(tf.random_normal([64])), 
                   'b_conv2': tf.Variable(tf.random_normal([128])), 
                   'b_conv3': tf.Variable(tf.random_normal([256])),
                   'b_fc': tf.Variable(tf.random_normal([256])), 
                   'b_out': tf.Variable(tf.random_normal([par['n_output']]))}

        # print("Original: ", par['inp_img_shape'])
        # print("Input: ", self.input_data)
        x = tf.reshape(self.input_data, shape=[par['batch_train_size'],*par['inp_img_shape'],1])
        # print("X: ", x.shape)
        # print()

        # might need to consider relu
        # conv1 = self.conv2d(x, weights['W_conv1'])
        # print("Conv1: ", conv1.shape)
        # conv1 = self.max_pool2d(conv1) + biases['b_conv1']
        # print("Conv1: ", conv1.shape)
        # print()
        conv1    = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
        conv2    = tf.layers.conv2d(inputs=maxpool1, filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
        conv3    = tf.layers.conv2d(inputs=maxpool2, filters=256, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
        # print(conv1.shape, maxpool1.shape)
        # print(conv2.shape, maxpool2.shape)
        # print(conv3.shape, maxpool3.shape)

        latent = tf.image.resize_images(maxpool3, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv4 = tf.layers.conv2d(inputs=latent, filters=256, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        upsample2 = tf.image.resize_images(conv4, size=(64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv5 = tf.layers.conv2d(inputs=upsample2, filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        upsample3 = tf.image.resize_images(conv5, size=(128,128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv6 = tf.layers.conv2d(inputs=upsample3, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        # print(conv6.shape)

        logits = tf.layers.conv2d(inputs=conv6, filters=3, kernel_size=(3,3), padding='same', activation=None)
        # print(logits.shape)
        self.output = tf.nn.relu(tf.reshape(logits, [par['batch_train_size'],par['n_output']]))

        # conv2 = self.conv2d(conv1, weights['W_conv2'])
        # print("Conv2: ", conv2.shape)
        # conv2 = self.max_pool2d(conv2) + biases['b_conv2']
        # print("Conv2: ", conv2.shape)
        # print()


        # conv3 = self.conv2d(conv2, weights['W_conv3'])
        # print("Conv3: ", conv3.shape)
        # conv3 = self.max_pool2d(conv3) + biases['b_conv3']
        # print("Conv3: ", conv3.shape)
        # print()

        # latent = tf.reshape(conv3, [par['batch_train_size'],conv3.shape[1]*conv3.shape[2]*conv3.shape[3]])
        # latent = tf.nn.relu(tf.matmul(latent, weights['W_fc']) + biases['b_fc'])
        # print("Latent: ", latent)

        # print(latent.shape) #32 x 256
        # fc = tf.reshape(conv2, [-1,7,7,64])
        # fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

        # self.output = tf.matmul(latent, weights['W_out']) + biases['b_out']
        # print("Output: ", self.output)
        



        """
        
        # Get weights and biases
        if par['num_layers'] == 5:
            with tf.variable_scope('encoder'):
                self.W_in = tf.get_variable('W_in', initializer=par['W_in_init'], trainable=True)
                self.b_enc = tf.get_variable('b_enc', initializer=par['b_enc_init'], trainable=True)
                self.W_enc = tf.get_variable('W_enc', initializer=par['W_enc_init'], trainable=True)
                self.b_latent = tf.get_variable('b_latent', initializer=par['b_latent_init'], trainable=True)
                self.W_link = tf.get_variable('W_link', initializer=par['W_link_init'], trainable=True)
                self.b_link = tf.get_variable('b_link', initializer=par['b_link_init'], trainable=True)

            with tf.variable_scope('decoder'):
                self.W_dec = tf.get_variable('W_dec', initializer=par['W_dec_init'], trainable=True)
                self.b_dec = tf.get_variable('b_dec', initializer=par['b_dec_init'], trainable=True)
                self.W_link2 = tf.get_variable('W_link2', initializer=par['W_link2_init'], trainable=True)
                self.b_link2 = tf.get_variable('b_link2', initializer=par['b_link2_init'], trainable=True)
                self.W_out = tf.get_variable('W_out', initializer=par['W_out_init'], trainable=True)
                self.b_out = tf.get_variable('b_out', initializer=par['b_out_init'], trainable=True)
        
        elif par['num_layers'] == 3:
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

        elif par['num_layers'] == 2:
            with tf.variable_scope('encoder'):
                self.W_in = tf.get_variable('W_in', initializer=par['W_in_init'], trainable=True)
                self.b_enc = tf.get_variable('b_enc', initializer=par['b_enc_init'], trainable=True)

            with tf.variable_scope('decoder'):
                self.W_dec = tf.get_variable('W_dec', initializer=par['W_dec_init'], trainable=True)
                self.b_dec = tf.get_variable('b_dec', initializer=par['b_dec_init'], trainable=True)
                self.W_out = tf.get_variable('W_out', initializer=par['W_out_init'], trainable=True)
                self.b_out = tf.get_variable('b_out', initializer=par['b_out_init'], trainable=True)

        # Run input through the model layers
        if par['num_layers'] == 5:
            self.enc = tf.nn.sigmoid(tf.add(tf.matmul(self.input_data, self.W_in), self.b_enc))
            self.link = tf.nn.sigmoid(tf.add(tf.matmul(self.enc, self.W_enc), self.b_latent))
            self.latent = tf.nn.sigmoid(tf.add(tf.matmul(self.link, self.W_link), self.b_link))
            self.link2 = tf.nn.sigmoid(tf.add(tf.matmul(self.latent, self.W_dec), self.b_dec))
            self.dec = tf.nn.sigmoid(tf.add(tf.matmul(self.link2, self.W_link2), self.b_link2))
            self.output = tf.nn.relu(tf.add(tf.matmul(self.dec, self.W_out), self.b_out))
        
        elif par['num_layers'] == 3:
            print("Relu with sigmoid")
            self.enc = tf.nn.relu(tf.add(tf.matmul(self.input_data, self.W_in), self.b_enc))
            self.latent = tf.nn.relu(tf.add(tf.matmul(self.enc, self.W_enc), self.b_latent))
            self.dec = tf.nn.relu(tf.add(tf.matmul(self.latent, self.W_dec), self.b_dec))
            self.output = tf.nn.relu(tf.add(tf.matmul(self.dec, self.W_out), self.b_out))
        
        elif par['num_layers'] == 2:
            self.enc = tf.nn.sigmoid(tf.add(tf.matmul(self.input_data, self.W_in), self.b_enc))
            self.dec = tf.nn.sigmoid(tf.add(tf.matmul(self.enc, self.W_dec), self.b_dec))
            self.output = tf.nn.relu(tf.add(tf.matmul(self.dec, self.W_out), self.b_out))
        """

    def optimize(self):
        # Calculae loss
        # self.loss = tf.reduce_mean(tf.square(self.target_data - self.output))
        self.loss = tf.losses.mean_squared_error(self.target_data, self.output)
        self.train_op = tf.train.AdamOptimizer(par['learning_rate']).minimize(self.loss)


def main(gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Generate stimulus
    stim = Stimulus()

    # Placeholders for the tensorflow model
    x = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_input']])
    y = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_output']])
    
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

        # Train the model
        start = time.time()
        for i in range(par['num_iterations']):

            # Generate training set
            input_data, target_data = stim.generate_train_batch()
            feed_dict = {x: input_data, y: target_data}
            _, train_loss, model_output = sess.run([model.train_op, model.loss, model.output], feed_dict=feed_dict)
            
            # if np.max(model_output) > 255 or np.min(model_output) <= 0:
                # print("Input:", round(np.max(input_data),3), round(np.min(input_data),3))
                # print("Output:", round(np.max(model_output),3), round(np.min(model_output),3))

            # Check current status
            if i % par['print_iter'] == 0:

                # Print current status
                print('Model {:2} | Task: {:s} | Iter: {:6} | Loss: {:8.3f} | Run Time: {:5.3f}s'.format( \
                    par['run_number'], par['task'], i, train_loss, time.time()-start))
                losses.append(train_loss)

                # Save one training and output img from this iteration
                if i % par['save_iter'] == 0:

                    # Generate batch from testing set and check the output
                    test_input, test_target = stim.generate_test_batch()
                    feed_dict = {x: test_input, y: test_target}
                    test_loss, test_output = sess.run([model.loss, model.output], feed_dict=feed_dict)
                    testing_losses.append(test_loss)

                    # Results from a training sample
                    original1 = target_data[0].reshape(par['out_img_shape'])
                    output1 = model_output[0].reshape(par['out_img_shape'])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # cv2.putText(original1,'Training',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
                    # cv2.putText(output1,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)

                    # Results from a testing sample
                    original2 = test_target[1].reshape(par['out_img_shape'])
                    output2 = test_output[1].reshape(par['out_img_shape'])
                    original3 = test_target[2].reshape(par['out_img_shape'])
                    output3 = test_output[2].reshape(par['out_img_shape'])
                    # cv2.putText(original2,'Testing',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
                    # cv2.putText(output2,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
                
                    vis1 = np.concatenate((original1, output1), axis=1)
                    vis2 = np.concatenate((original2, output2), axis=1)
                    vis3 = np.concatenate((original3, output3), axis=1)
                    vis = np.concatenate((vis1, vis2), axis=0)
                    vis = np.concatenate((vis, vis3), axis=0)
                    if par['normalize01']:
                        print("UN-NORMALIZE")
                        if np.max(vis) > 1 or np.min(vis) < 0:
                            print(np.max(vis))
                            print(np.min(vis))
                            print("Something is wrong")
                            quit()
                        vis *= 255
                        # vis = np.int8(vis)

                    cv2.imwrite(par['save_dir']+'run_'+str(par['run_number'])+'_test_'+str(i)+'.png', vis)

                    # weights = eval_weights()
                    weights = None
                    pickle.dump({'weights': weights, 'losses': losses, 'test_loss': testing_losses, 'last_iter': i}, \
                        open(par['save_dir']+'run_'+str(par['run_number'])+'_model_stats.pkl', 'wb'))


                # Plot loss curve
                if i > 0:
                    plt.plot(losses[1:])
                    plt.savefig(par['save_dir']+'run_'+str(par['run_number'])+'_training_curve.png')
                    plt.close()

                # Stop training if loss is small enough (just for sweep purposes)
                # if train_loss < 100:
                    # break

def eval_weights():

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











