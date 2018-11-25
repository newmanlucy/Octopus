import copy
import time
import cv2
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.saved_model import tag_constants
from stimulus import Stimulus
from parameters import *

# Ignore tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


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

    cv2.imwrite(par['save_dir']+'simulation_'+str(par['run_number'])+'_output_'+str(i)+'.png', vis)


def main(gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Generate stimulus
    stim = Stimulus()
    x = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_input']])
    y = tf.placeholder(tf.float32, shape=[par['batch_train_size'],par['n_output']])

    # Model stats
    losses = []
    testing_losses = []

    tf.reset_default_graph()
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            imported_graph = tf.train.import_meta_graph('conv_model.meta', clear_devices=True)
            imported_graph.restore(sess, tf.train.latest_checkpoint('./'))            
            
        for i in range(10):
            
            # Generate batch from testing set and check the output
            test_input, test_target = stim.generate_test_batch()
            feed_dict = {'x:0': test_input, 'y:0': test_target}
            test_loss, test_output = sess.run(['l:0', 'o:0'], feed_dict=feed_dict)
            print("FINAL TEST LOSS IS: ", test_loss)
            plot_testing(test_target, test_output, i)


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
            'a_note'            : 'testing save and load model conv',
            'input_dir'         : './bw_im2/',
            'target_dir'        : './raw_im2/',
            'batch_train_size'  : 32,
            'learning_rate'     : 0.001,
            'normalize01'       : False,
            'num_layers'        : 3,
            'run_number'        : 2,
            "save_iter"         : 2000,
            'task'              : 'bw1_to_color'
        }
        update_parameters(updates)
        print('Model number ' + str(par['run_number']) + ' running!')
        main(gpu_id)
        print('Model run concluded.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))


