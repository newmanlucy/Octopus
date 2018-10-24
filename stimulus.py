import numpy as np
import cv2
import os
from parameters import *

class Stimulus:

    def __init__(self):

        # Get files from img_dir
        files = os.listdir(par['img_dir'])
        for f in files:
            if 'clean' not in f:
                files.remove(f)

        # Separate Training and Testing Data
        idx = round(len(files)*0.8)
        training_files = [os.path.join(par['img_dir'], f) for f in files[:idx]]
        testing_files = [os.path.join(par['img_dir'], f) for f in files[idx:]]

        # Load up images from the files
        # 'bw_to_bw': going from three channel bw image to three channel bw image
        # 'bw_to_bw_simple': going from one channel bw image to one channel bw image
        if par['task'] == 'bw_to_bw':
            training_imgs = [cv2.imread(f) for f in training_files]
            testing_imgs = [cv2.imread(f) for f in testing_files]
        elif par['task'] == 'bw_to_bw_simple':
            training_imgs = [cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY) for f in training_files]
            testing_imgs = [cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY) for f in testing_files]
        else:
            raise Exception('Task "{}" not yet implemented.'.format(par['task']))

        # Resize the images to desired size
        training_imgs_small = [cv2.resize(img, par['img_shape'][0:2]) for img in training_imgs]
        testing_imgs_small = [cv2.resize(img, par['img_shape'][0:2]) for img in testing_imgs]
        
        # Reshape the images to input dimensions
        self.training_data = np.array([img.reshape(par['n_input']) for img in training_imgs_small])
        self.testing_data = np.array([img.reshape(par['n_input']) for img in testing_imgs_small])

        vis = self.training_data[0].reshape(par['img_shape'])
        cv2.imwrite('./debug.png', vis)


    def generate_train_batch(self):

        # Pick images to be used for training from training data set
        idx = np.random.choice(len(self.training_data), size=par['batch_train_size'], replace=False)
        input_data = self.training_data[idx]
        target_data = self.training_data[idx]

        return input_data, target_data

    def generate_test_batch(self):

        # Pick images to be used for training from training data set
        idx = np.random.choice(len(self.testing_data), size=par['batch_train_size'])
        input_data = self.testing_data[idx]
        target_data = self.testing_data[idx]

        return input_data, target_data

if __name__ == "__main__":
    stim = Stimulus()
    stim.generate_train_batch()