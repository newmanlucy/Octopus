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
from best_evo import EvoModel

# Ignore tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def load_model():

    # Reset Tensorflow graph
    tf.reset_default_graph()

    # Generate stimulus
    stim = Stimulus()
    evo_model = EvoModel()

    saved_evo_model = pickle.load(open('./savedir/conv_task/run_14_model_stats.pkl','rb'))
    best_weights = {}
    for key, val in saved_evo_model['var_dict'].items():
        best_weights[key] = val[0]
    evo_model.update_variables(best_weights)
    print('Loaded evo model')

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    folder = './latent_all_img_batch16_filt16_loss150/'
    conv_model = tf.train.import_meta_graph(folder + 'conv_model_with_latent.meta', clear_devices=True)
    conv_model.restore(sess, tf.train.latest_checkpoint(folder)) 
    print('Loaded conv model from',folder)

    return stim, evo_model, sess

def run_model(stim, evo_model, sess, input_data, conv_target, evo_target):    

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
    evo_model.judge_models()

    evo_output = evo_model.output
    test_loss = evo_model.get_losses(True)
    print('Evo loss:',test_loss)
    print((evo_output.shape, conv_output.shape))
    return evo_output
                # end = time.time()
                # print(end-start)
            # display evo_output[0]

def actually_run_model(stim, evo_model, sess, input_data, conv_target, evo_target):
    # Run Model
    t0 = time.time()
    try:
        # print('Model number ' + str(par['run_number']) + ' running!')
        evo_output = run_model(stim, evo_model, sess, input_data, conv_target, evo_target)
        print('Model run concluded.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))
        return evo_output
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))

SMALL = 128
BIG = 480
SLEEP = .5
NUM = 16

def get_video_and_run():
    cap = cv2.VideoCapture(0)
    updates = {
        'one_img'           : False,
        'batch_train_size'  : 16,
        'num_conv1_filters' : 16,
        'task'              : 'conv_task'
    }
    update_parameters(updates)
    stim, evo_model, sess = load_model()

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
        # cv2.imshow('frame', gray)
        grey = gray.reshape(SMALL * SMALL)
        gray = [grey] * NUM

        grey_3chan = gray_3chan.reshape(SMALL * SMALL * 3)
        gray_3chan = [grey_3chan] * NUM

        freme = frame.reshape(SMALL * SMALL * 3)
        frame = [freme] * NUM

        evo_output = actually_run_model(stim, evo_model, sess, gray, gray_3chan, frame)[0]
        evo_output = evo_output.reshape((SMALL,SMALL,3))
        # Display the resulting frame
        evo_output = cv2.resize(evo_output, (BIG, BIG))
        print("evo_output after resize")
        print(evo_output.shape)
        cv2.imshow('frame',evo_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # break

        # sleep(SLEEP)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # actually_run_model(None,None,None)
    get_video_and_run()