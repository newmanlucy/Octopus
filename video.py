import numpy as np
import cv2
from time import sleep

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

def run_model(input_data, conv_target, evo_target, gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        gpu_id = None

    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Generate stimulus
    stim = Stimulus()
    evo_model = EvoModel()

    saved_evo_model = pickle.load(open('./run_14_model_stats.pkl','rb'))
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
            folder = './'
            conv_model = tf.train.import_meta_graph(folder + 'conv_model_with_latent.meta', clear_devices=True)
            conv_model.restore(sess, tf.train.latest_checkpoint(folder)) 
            print('Loaded conv model from',folder)

            # Generate batch from testing set and check the output
            # (16,128,128), (16,128,128,3) bw, (16,128,128,3) color
            input_data, conv_target, evo_target = stim.generate_test_batch()
            # for i in range(10):
                # start = time.time()
            feed_dict = {'x:0': input_data, 'y:0': conv_target}
            test_loss, conv_output, encoded = sess.run(['l:0', 'o:0','encoded:0'], feed_dict=feed_dict)

            # "TEST" EVO MODEL
            evo_model.load_batch(encoded, evo_target)
            evo_model.run_models()
            # evo_model.judge_models()

            evo_output = evo_model.output
            test_loss = evo_model.get_losses(True)
            testing_losses.append(test_loss[0])
    return evo_output
                # end = time.time()
                # print(end-start)
            # display evo_output[0]


def actually_run_model(input_data, conv_target, evo_target):
    # Run Model
    t0 = time.time()
    try:
        updates = {
            'a_note'            : 'latent to evo, all img',
            'input_dir'         : './bw_im/',
            'target_dir'        : './raw_im/',
            'print_iter'        : 1,
            'save_iter'         : 5,
            'batch_train_size'  : 16,
            'run_number'        : 14,
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
        run_model(input_data, conv_target, evo_target)
        print('Model run concluded.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))

SMALL = 128
BIG = 480
SLEEP = .5

def get_video_and_run():
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        h, w = frame.shape[:2]
        frame = cv2.resize(frame, ((w * SMALL) // h, SMALL))
        new_w = frame.shape[1]
        off = (new_w - SMALL) // 2
        frame = np.array([row[off:new_w-off] for row in frame])

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_3chan = np.array([[[x,x,x] for x in row] for row in gray])

        # (16,128,128), (16,128,128,3) bw, (16,128,128,3) color
        cv2.imshow('frame', gray)
        # gray = [gray] * 16
        # gray_3chan = [gray_3chan] * 16
        # frame = [frame] * 16
        # evo_output = actually_run_model(gray, gray_3chan, frame)
        # Display the resulting frame
        # evo_output = cv2.resize(evo_output, (BIG, BIG))
        # cv2.imshow('frame',evo_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        sleep(SLEEP)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # actually_run_model(None,None,None)
    get_video_and_run()