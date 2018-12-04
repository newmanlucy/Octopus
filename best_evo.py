"""
Catherine (Chaihyun) Lee & Lucy Newman (2018)
Contributions from Rick Stevens
"""
from evo_utils import *

class EvoModel:

    def __init__(self):
        
        self.make_constants()
        self.original = True
        self.var_dict = {}
 
    def update_variables(self, updates):
        for key, val in updates.items():
            self.var_dict[key] = to_gpu(val)

    def make_constants(self):
        constants = ['n_networks','mutation_rate','mutation_strength','cross_rate']

        self.con_dict = {}
        for c in constants:
            self.con_dict[c] = to_gpu(par[c])

    def load_batch(self, input_data, target_data):
        self.input_data = to_gpu(input_data)
        self.target_data = to_gpu(target_data)

    def run_models(self):
        conv1 = relu(convolve(self.input_data, self.var_dict, 'conv2_filter') + self.var_dict['conv2_bias'])
        self.output = cp.reshape(conv1, (par['batch_train_size'],*par['out_img_shape']))

    def judge_models(self):
        img_len = par['img_size']
        self.target_data = cp.reshape(self.target_data, (par['batch_train_size'],*par['out_img_shape']))
        trimmed_img = self.target_data[:,1:img_len-1,1:img_len-1,:]
        self.loss = cp.mean(cp.square(trimmed_img - self.output[:,1:img_len-1,1:img_len-1,:])).astype(cp.float64)
        self.output = cp.reshape(self.output, (par['batch_train_size'],par['n_output']))

    def get_losses(self, ranked=True):
        return to_cpu(self.loss)














