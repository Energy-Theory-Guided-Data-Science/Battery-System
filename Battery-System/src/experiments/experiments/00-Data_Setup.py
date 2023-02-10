#!/usr/bin/env python
# coding: utf-8

# # Create Data Setup
# This notebook serves the purpose of setting some general hyperparameters and defining training and test datasets. All of these parameters are saved to the folder Battery-System/models/training_setup from which all models have access. 

# In[16]:


import context
import numpy as np


# ### Set General Hyperparameters

# In[17]:


# set general hyperparamters which are required by all models
HYPER_PARAMS = {
    'd_sample': 2,                      # subsampling ratio
    'gauss_sigma': 10,                  # smoothing factor
    'stack': 0,                         # battery stack used to extract the data from
    'cell': 4,                          # specific battery cell to extract the data from
    'n_steps': 100,                     # defines M in the M-to-1 LSTM structure
    
    'n_lstm_units_1': 50,               # number of LSTM units in the first layer
    'alpha_1': 0.1,                     # alpha value for LeakyReLU acitvation function
    'n_lstm_units_2': 20,               # number of LSTM units in the second layer
    'alpha_2': 0.1,                     # alpha value for LeakyReLU activation function
    
    'activation_output_layer': 'tanh',  # output activation function
    'n_epochs': 10,                      # number of training epochs
    'optimizer': 'Adam',                # optimizer for model training
    'metric': 'mae',                    # performance metric during training
    'num_repeat': 10,                    # repetition factor for each training profile to create more auxiliary data

    
    'feature_range_cur_low': -1,        # lower bound of current input feature after scaling
    'feature_range_cur_high': 1,        # upper -||-
    'feature_range_charge_low': -1,     # lower bound of charge input feature after scaling
    'feature_range_charge_high': 1,     # upper -||-
    'feature_range_volt_low': -1,       # lower bound of voltage label after scaling
    'feature_range_volt_high': 1,       # upper -||-
    
    'boundary_cur_low': -10,            # lower bound of current value range in A
    'boundary_cur_high': 10,            # upper -||-
    'boundary_charge_low': -33.2,       # lower bound of charge value range in Ah
    'boundary_charge_high': 33.2,       # upper -||-
    'boundary_voltage_low': 3.304,      # lower bound of voltage value range in V
    'boundary_voltage_high': 3.982,     # upper -||-
}

# save hyperparameters
np.save('../../../models/training_setup/hyperparameters', HYPER_PARAMS)


# ### Define Training Datasets

# In[18]:


# initialize different sets of training data identifyed by their unique keys
TRAINING_SETS = {
    '10A_one': ['Profile 10A'],
    
    '10A_all': ['Profile 10A',
                'Profile 10A Run 040618',
                'Profile 10A Run 080618', 
                'Profile 10A Run 070618_3',
                'Profile 10A Run 070618',
                'Profile 10A Run 070618_2'],
    
    '-10A_all': ['Profile -10A',
                 'Profile -10A Run 070618',
                 'Profile -10A Run 070618_2',
                 'Profile -10A Run 070618_3',
                 'Profile -10A Run 080618_2',
                 'Profile -10A Run 080618_3'
                ],
    
    '25A_all': ['Profile 25A Run 2',
                'Profile 25A Run 040618',
                'Profile 25A Run 070618',
                'Profile 25A Run 070618_3',
                'Profile 25A Run 070618_4'
               ],
    
    '-25A_all': ['Profile -25A',
                 'Profile -25A Run 070618',
                 'Profile -25A Run 070618_2',
                 'Profile -25A Run 070618_3',
                 'Profile -25A Run 080618_2'
                ],
}

# save training data sets
np.save('../../../models/training_setup/training_sets', TRAINING_SETS)


# ### Define Test Datasets

# In[19]:


# initialize different sets of test data identifyed by the unique keys
TEST_SETS = {
    'Use_Case_1': ['profile_-10A_25A_19_11_18'],
    'Use_Case_2': ['profile_-25A_10A_04_12_18'],
    'Use_Case_3': ['osc_19_11_18'],
    
    'Untitled_1':['Profile 10A Run 070618_2'],
    'Untitled_2':['Profile 10A'],
    'Untitled_3':['Profile 10A Run 070618_2'],
    'Untitled_4':['Profile 25A Run 2'],
    
    'Reproduction': ['Profile 10A'],
    'Abstraction': ['Profile 10A 3x'],
    'Generalization': ['Profile -10A'],
}

# save test data sets
np.save('../../../models/training_setup/test_sets', TEST_SETS)


# In[ ]:




