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
from latent_model import EvoModel

# Ignore tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main(gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        gpu_id = None

    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Generate stimulus
    stim = Stimulus()
    evo_model = EvoModel()

    saved_evo_model = pickle.load(open('./savedir/conv_task/run_21_model_stats.pkl','rb'))
    evo_model.update_variables(saved_evo_model['var_dict'])
    print('Loaded evo model')

    # Model stats
    losses = []
    testing_losses = []

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            folder = './latent_all_img_batch16_filt16_loss80/'
            conv_model = tf.train.import_meta_graph(folder + 'conv_model_with_latent.meta', clear_devices=True)
            conv_model.restore(sess, tf.train.latest_checkpoint(folder)) 
            print('Loaded conv model from',folder)

            # Generate batch from testing set and check the output
            # bw(16,128,128), bw(16,128,128,3), color(16,128,128,3)
            for i in range(2):
                input_data, conv_target, evo_target = stim.generate_train_batch()
                start = time.time()
                feed_dict = {'x:0': input_data, 'y:0': conv_target}
                test_loss, conv_output, encoded = sess.run(['l:0', 'o:0','encoded:0'], feed_dict=feed_dict)

                # "TEST" EVO MODEL
                evo_model.load_batch(encoded, evo_target)
                evo_model.run_models()
                evo_model.judge_models()

                evo_output = evo_model.output
                test_loss = evo_model.get_losses(True)
                testing_losses.append(test_loss[0])
                end = time.time()
                print('Time:', end-start)

                plot_outputs(conv_target, conv_output, evo_target, evo_output, i)


def plot_outputs(target_data, model_output, test_target, test_output, i):

    # Results from a training sample
    for j in range(4):
        outputs = []
        for b in range(4):
            batch = j*4 + b
            original1 = target_data[batch].reshape(par['out_img_shape'])
            output1 = model_output[batch].reshape(par['out_img_shape'])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(original1,'Conv',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
            cv2.putText(output1,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)

            # Results from a testing sample
            original2 = test_target[batch].reshape(par['out_img_shape'])
            output2 = test_output[0][batch].reshape(par['out_img_shape'])
            output3 = test_output[1][batch].reshape(par['out_img_shape'])
            output4 = test_output[2][batch].reshape(par['out_img_shape'])
            cv2.putText(original2,'Evo',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
            cv2.putText(output2,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
            cv2.putText(output3,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)
            cv2.putText(output4,'Output',(5,20), font, 0.5,(255,255,255), 2, cv2.LINE_AA)

            vis1 = np.concatenate((original1, output1), axis=1)
            vis2 = np.concatenate((original2, output2), axis=1)
            vis3 = np.concatenate((original2, output3), axis=1)
            vis4 = np.concatenate((original2, output4), axis=1)
            vis = np.concatenate((vis1, vis2), axis=0)
            vis = np.concatenate((vis, vis3), axis=0)
            vis = np.concatenate((vis, vis4), axis=0)
            outputs.append(vis)

        if len(outputs) == 1:
            vis = outputs[0]
        else:
            vis = np.concatenate((outputs[0],outputs[1]), axis=1)
            for batch in range(2,len(outputs)):
                vis = np.concatenate((vis,outputs[batch]), axis=1)
        
        cv2.imwrite('{}testing_run{}_{}_{}.png'.format(par['save_dir'],par['run_number'],j,i),vis)


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
            'a_note'            : 'latent to evo, all img',
            'print_iter'        : 1,
            'loss_count'        : 16,
            'save_iter'         : 5,
            'batch_train_size'  : 16,
            'run_number'        : 0,
            'num_conv1_filters' : 16,
            'n_networks'        : 65,
            'survival_rate'     : 0.12,
            'mutation_rate'     : 0.6,
            'mutation_strength' : 0.45,
            'migration_rate'    : 0.1,
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







