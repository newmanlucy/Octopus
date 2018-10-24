"""
Catherine (Chaihyun) Lee & Lucy Newman (2018)
Contributions from Rick Stevens
"""

import tensorflow as tf
import cv2
import time
import numpy as numpy
from stimulus import Stimulus
import matplotlib.pyplot as plt
from parameters import *

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
            self.enc = tf.nn.sigmoid(tf.add(tf.matmul(self.input_data, self.W_in), self.b_enc))
            self.latent = tf.nn.sigmoid(tf.add(tf.matmul(self.enc, self.W_enc), self.b_latent))
            self.dec = tf.nn.sigmoid(tf.add(tf.matmul(self.latent, self.W_dec), self.b_dec))
            self.output = tf.nn.relu(tf.add(tf.matmul(self.dec, self.W_out), self.b_out))
        
        elif par['num_layers'] == 2:
            self.enc = tf.nn.sigmoid(tf.add(tf.matmul(self.input_data, self.W_in), self.b_enc))
            self.dec = tf.nn.sigmoid(tf.add(tf.matmul(self.enc, self.W_dec), self.b_dec))
            self.output = tf.nn.relu(tf.add(tf.matmul(self.dec, self.W_out), self.b_out))

        # # LITERATURE STUFF
        # if par['use_literature_code']:
        #     n_nodes_inpl = 128*128
        #     n_nodes_hl1 = par['n_hidden']
        #     n_nodes_hl2 = par['n_hidden']
        #     n_nodes_outl = 128*128

        #     hidden_1_layer_vals = {
        #       'weights':tf.Variable(tf.random_normal([n_nodes_inpl,n_nodes_hl1])),
        #       'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))  }
                
        #     # second hidden layer has 32*32 weights and 32 biases
        #     hidden_2_layer_vals = {
        #       'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        #       'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))  }
        
        #     # second hidden layer has 32*784 weights and 784 biases
        #     output_layer_vals = {
        #       'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_outl])),
        #         'biases':tf.Variable(tf.random_normal([n_nodes_outl])) }

            
        #     layer1 = tf.nn.sigmoid(tf.add(tf.matmul(self.input_data, hidden_1_layer_vals['weights']), hidden_1_layer_vals['biases']))
        #     layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, hidden_2_layer_vals['weights']), hidden_2_layer_vals['biases']))
        #     if par['normalize']:
        #         output_layer = tf.matmul(layer2, output_layer_vals['weights']) + output_layer_vals['biases']
        #     else:
        #         output_layer = tf.nn.relu(tf.add(tf.matmul(layer2, output_layer_vals['weights']), output_layer_vals['biases']))
        #     self.output = output_layer

    def optimize(self):
        # Calculae loss
        self.loss = tf.reduce_mean(tf.square(self.target_data - self.output))
        self.train_op = tf.train.AdagradOptimizer(par['learning_rate']).minimize(self.loss)


def main():

    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Generate stimulus
    stim = Stimulus()

    # Placeholders for the tensorflow model
    x = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_input']])
    y = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_output']])
    
    # Model stats
    losses = []

    with tf.Session() as sess:

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

            # Check current status
            if i % par['print_iter'] == 0:

                # Print current status
                print('Iter: {:8} | Loss: {:8.3f} | Run Time: {:5.3f}s'.format(i, train_loss, time.time()-start))
                losses.append(train_loss)

                # Save one training and output img from this iteration
                if i % par['save_iter'] == 0:
                    original1 = input_data[0].reshape(par['img_shape'])
                    output1 = model_output[0].reshape(par['img_shape'])
                    original2 = input_data[1].reshape(par['img_shape'])
                    output2 = model_output[1].reshape(par['img_shape'])
                    original3 = input_data[2].reshape(par['img_shape'])
                    output3 = model_output[2].reshape(par['img_shape'])
                
                    vis1 = np.concatenate((original1, output1), axis=1)
                    vis2 = np.concatenate((original2, output2), axis=1)
                    vis3 = np.concatenate((original3, output3), axis=1)
                    vis = np.concatenate((vis1, vis2), axis=0)
                    vis = np.concatenate((vis, vis3), axis=0)
                    cv2.imwrite(par['save_dir']+'run_'+str(par['run_number'])+'_test_'+str(i)+'.png', vis)

                # Plot loss curve
                if i > 0:
                    plt.plot(losses[1:])
                    plt.savefig(par['save_dir']+'run_'+str(par['run_number'])+'_training_curve.png')

                # Stop training if loss is small enough (just for sweep purposes)
                # if train_loss < 100:
                    # break

if __name__ == "__main__":
    main()











