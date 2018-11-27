import numpy as np
import cv2
import os
from parameters import *

class Stimulus:

    def __init__(self):

        # Get files from img_dir
        files = os.listdir(par['input_dir'])
        for f in files:
            if 'clean' not in f:
                files.remove(f)

        # Separate Training and Testing Data
        idx = round(len(files)*0.8)
        training_input_files  = [os.path.join(par['input_dir'], f) for f in files[:idx]]
        training_target_files = [os.path.join(par['target_dir'], f) for f in files[:idx]]
        testing_input_files   = [os.path.join(par['input_dir'], f) for f in files[idx:]]
        testing_target_files  = [os.path.join(par['target_dir'], f) for f in files[idx:]]

        # Load up images from the files
        # 'bw_to_bw': going from three channel bw image to three channel bw image
        # 'bw_to_bw_simple': going from one channel bw image to one channel bw image
        if par['task'] == 'bw_to_bw' or par['task'] == 'bw3_to_color':
            training_input_imgs  = [cv2.imread(f) for f in training_input_files]
            training_target_imgs = [cv2.imread(f) for f in training_target_files]
            testing_input_imgs   = [cv2.imread(f) for f in testing_input_files]
            testing_target_imgs  = [cv2.imread(f) for f in testing_target_files]

        elif par['task'] == 'bw_to_bw_simple':
            training_input_imgs  = [cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY) for f in training_input_files]
            training_target_imgs = [cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY) for f in training_target_files]
            testing_input_imgs   = [cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY) for f in testing_input_files]
            testing_target_imgs  = [cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY) for f in testing_target_files]
        
        elif par['task'] == 'bw1_to_color':
            training_input_imgs  = [cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY) for f in training_input_files]
            training_target_imgs = [cv2.imread(f) for f in training_target_files]
            testing_input_imgs   = [cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY) for f in testing_input_files]
            testing_target_imgs  = [cv2.imread(f) for f in testing_target_files]

        elif par['task'] == 'conv_task':
            # bw1 input for conv model / bw3 target for conv model
            # bw3 input for evo model / color3 target for evo model
            training_input_imgs   = [cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY) for f in training_input_files]
            training_target_imgs1 = [cv2.imread(f) for f in training_input_files]
            training_target_imgs2 = [cv2.imread(f) for f in training_target_files]
            
            testing_input_imgs    = [cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY) for f in testing_input_files]
            testing_target_imgs1  = [cv2.imread(f) for f in testing_input_files]
            testing_target_imgs2  = [cv2.imread(f) for f in testing_target_files]


        else:
            raise Exception('Task "{}" not yet implemented.'.format(par['task']))

        # Resize the images to desired size
        training_imgs_small      = [cv2.resize(img, par['inp_img_shape'][0:2]) for img in training_input_imgs]
        training_imgs_small_out1 = [cv2.resize(img, par['out_img_shape'][0:2]) for img in training_target_imgs1]
        training_imgs_small_out2 = [cv2.resize(img, par['out_img_shape'][0:2]) for img in training_target_imgs2]
        testing_imgs_small       = [cv2.resize(img, par['inp_img_shape'][0:2]) for img in testing_input_imgs]
        testing_imgs_small_out1  = [cv2.resize(img, par['out_img_shape'][0:2]) for img in testing_target_imgs1]
        testing_imgs_small_out2  = [cv2.resize(img, par['out_img_shape'][0:2]) for img in testing_target_imgs2]

        
        # Reshape the images to input dimensions
        self.training_data    = np.array([np.array(img).reshape(par['n_input']) for img in training_imgs_small])
        self.training_output1 = np.array([np.array(img).reshape(par['n_output']) for img in training_imgs_small_out1])
        self.training_output2 = np.array([np.array(img).reshape(par['n_output']) for img in training_imgs_small_out2])
        self.testing_data     = np.array([np.array(img).reshape(par['n_input']) for img in testing_imgs_small])
        self.testing_output1  = np.array([np.array(img).reshape(par['n_output']) for img in testing_imgs_small_out1])
        self.testing_output2  = np.array([np.array(img).reshape(par['n_output']) for img in testing_imgs_small_out2])
        

    def generate_train_batch(self):

        # Pick images to be used for training from training data set
        idx = np.random.choice(len(self.training_data), size=par['batch_train_size'], replace=False)
        input_data   = self.training_data[idx]
        target_data1 = self.training_output1[idx]
        target_data2 = self.training_output2[idx]

        # Checking input image
        vis = self.training_data[0].reshape(par['inp_img_shape'])
        cv2.imwrite(par['save_dir']+'debug_input.png', vis)

        # Checking target image
        vis = self.training_output1[0].reshape(par['out_img_shape'])
        cv2.imwrite(par['save_dir']+'debug_target1.png', vis)
        vis = self.training_output2[0].reshape(par['out_img_shape'])
        cv2.imwrite(par['save_dir']+'debug_target2.png', vis)

        return input_data, target_data1, target_data2

    def generate_test_batch(self):

        # Pick images to be used for training from training data set
        idx = np.random.choice(len(self.testing_data), size=par['batch_train_size'])
        input_data   = self.testing_data[idx]
        target_data1 = self.testing_output1[idx]
        target_data2 = self.testing_output2[idx]

        return input_data, target_data1, target_data2

if __name__ == "__main__":
    stim = Stimulus()
    stim.generate_train_batch()