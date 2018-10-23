"""
Catherine (Chaihyun) Lee & Lucy Newman (2018)
Contributions from Rick Stevens
"""

import tensorflow as tf
import cv2
import AdamOpt
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
		with tf.variable_scope('encoder'):
			self.W_in = tf.get_variable('W_in', initializer=par['W_in_init'], trainable=True)
			self.W_enc = tf.get_variable('W_enc', initializer=par['W_enc_init'], trainable=True)
			self.b_latent = tf.get_variable('b_latent', initializer=par['b_latent_init'], trainable=True)

		with tf.variable_scope('decoder'):
			self.W_dec = tf.get_variable('W_dec', initializer=par['W_dec_init'], trainable=True)
			self.b_dec = tf.get_variable('b_dec', initializer=par['b_dec_init'], trainable=True)
			self.W_out = tf.get_variable('W_out', initializer=par['W_out_init'], trainable=True)
			self.b_out = tf.get_variable('b_out', initializer=par['b_out_init'], trainable=True)

		# Run input through the model layers
		self.enc = tf.nn.relu(tf.matmul(self.input_data, self.W_in))
		self.latent = tf.nn.relu(tf.matmul(self.enc, self.W_enc) + self.b_latent)
		self.dec = tf.nn.relu(tf.matmul(self.latent, self.W_dec) + self.b_dec)
		self.output = tf.matmul(self.latent, self.W_out) + self.b_out


	def optimize(self):
		# Calculae loss
		self.loss = tf.reduce_mean(tf.square(self.target_data - self.output))

		# Use Adam Optimizer to train
		opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
		grads_and_vars = opt.compute_gradients(self.loss)
		self.train_op = opt.apply_gradients(grads_and_vars)

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
		for i in range(par['num_iterations']):

			# Generate training set
			input_data, target_data = stim.generate_train_batch()
			feed_dict = {x: input_data, y: target_data}
			_, train_loss, model_output = sess.run([model.train_op, model.loss, model.output], feed_dict=feed_dict)

			# Check current status
			if i % par['print_iter'] == 0:

				print('Iter: {:8} | Loss: {:8.3f}'.format(i, train_loss))
				losses.append(train_loss)

				# Save one training and output img from this iteration
				original = input_data[0].reshape(par['img_shape'])
				output = model_output[0].reshape(par['img_shape'])
				
				vis = np.concatenate((original, output), axis=1)
				cv2.imwrite(par['save_dir']+'run_'+str(par['run_number'])+'_test_'+str(i)+'.png', vis)

				# Stop training if loss is small enough (just for sweep purposes)
				if train_loss < 100:
					break

		# Plot loss curve on a log scale
		plt.plot(losses)
		plt.yscale('log')
		plt.savefig(par['save_dir']+'run_'+str(par['run_number'])+'_training_curve.png')

if __name__ == "__main__":
	main()











