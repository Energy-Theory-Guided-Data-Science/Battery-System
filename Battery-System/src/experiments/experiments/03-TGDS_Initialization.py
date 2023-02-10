#!/usr/bin/env python
# coding: utf-8

# # TGDS Model using Initialization

# In[1]:

import context
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import sys, os
from numba import cuda

import src.data.data_preprocessing as util
import src.models.pretrained_lstm_model as lstm

tf.compat.v1.set_random_seed(1)
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# ### Set Hyperparameters

# In[2]:


# load general hyperparameters
HYPER_PARAMS =  np.load('../../../models/training_setup/hyperparameters.npy', allow_pickle=True)
HYPER_PARAMS = HYPER_PARAMS.item()

# add aditional model-spefic hyperparameters
model_hyperparameters = {
    'n_pretraining_epochs': HYPER_PARAMS['n_epochs'],       # number of training epochs
    'n_training_epochs': HYPER_PARAMS['n_epochs'],          # number of training epochs
    'n_features': 2,                                        # number of input features
    'd_t': 0.25,                                            # current integration factor
    'num_repeat_pretrain': HYPER_PARAMS['num_repeat'], # repetition factor for each pretraining profile to create more auxiliary data
    'theory_model': 2228,                              # the unique thevenin model ID for which 
                                                       #the parameters have been computed
}

# update hyperparameters
HYPER_PARAMS.update(model_hyperparameters)


# ### Prepare Pretraining Data

# In[3]:


# load training sets dictionary
TRAINING_SETS =  np.load('../../../models/training_setup/training_sets.npy', allow_pickle=True)
TRAINING_SETS = TRAINING_SETS.item()

# select needed pretraining set
pretraining_sets = TRAINING_SETS['10A_all']

# create pretraining data
pretrain_data = []
for set in pretraining_sets:
    set_repeat = [set] * HYPER_PARAMS['num_repeat_pretrain']
    pretrain_data += set_repeat
pretrain_data = np.array(pretrain_data)

# prepare pretraining data
X_pretrain, y_pretrain, scalers_pretrain = util.prepare_pretraining_input(HYPER_PARAMS, pretrain_data, 0, 4)


# ### Prepare Training/Validation/Test Data

# In[4]:


# load training sets dictionary
TRAINING_SETS =  np.load('../../../models/training_setup/training_sets.npy', allow_pickle=True)
TRAINING_SETS = TRAINING_SETS.item()

# select correct training set
training_sets = TRAINING_SETS['10A_all']

# create training data
train_data = []
for set in training_sets:
    set_repeat = [set] * HYPER_PARAMS['num_repeat']
    train_data += set_repeat
train_data = np.array(train_data)

# select first profile for validation
validation_profile = [train_data[0]] 

# select arbitrary profile for testing
test_profile = np.random.choice(train_data, 1) 

# prepare input data
X_train, y_train, scalers_train = util.prepare_feature_engineering_input(HYPER_PARAMS, train_data, HYPER_PARAMS['stack'], HYPER_PARAMS['cell'])
X_validation, y_validation, _ = util.prepare_feature_engineering_input(HYPER_PARAMS, validation_profile, HYPER_PARAMS['stack'], HYPER_PARAMS['cell'])
X_test, y_test, _ = util.prepare_feature_engineering_input(HYPER_PARAMS, test_profile, HYPER_PARAMS['stack'], HYPER_PARAMS['cell'])


# ### Initialize Model

# In[5]:


lstm = lstm.Model()
lstm.initialize(HYPER_PARAMS)


# ### Pretrain Model

# In[6]:


_, fig = lstm.pretrain(X_pretrain, y_pretrain, scalers_train)

# save model and hyperparameters
MODEL_ID = str(np.random.randint(10000))

lstm.sequence_autoencoder.save('../../../models/TGDS/' + str(MODEL_ID))
np.save('../../../models/TGDS/' + str(MODEL_ID) + '/hyperparameters', HYPER_PARAMS)
fig.savefig('../../../reports/figures/theory_guided_pretraining-' + str(MODEL_ID) + '-learning_curve.png')


# ### Train Model

# In[7]:


_, fig = lstm.train(X_train, y_train, scalers_train)

# save model and hyperparameters
MODEL_ID = str(np.random.randint(10000))

lstm.model.save('../../../models/TGDS/' + str(MODEL_ID))
np.save('../../../models/TGDS/' + str(MODEL_ID) + '/hyperparameters', HYPER_PARAMS)
fig.savefig('../../../reports/figures/theory_guided_pretraining-' + str(MODEL_ID) + '-learning_curve.png')


# ### Test Model

# In[8]:


print('Validation Profile:', validation_profile)
print('Test Profile:', test_profile)

yhat_train_unscaled, _, _, fig = lstm.test(X_train, y_train, X_validation, y_validation, X_test, y_test, scalers_train)

# save plots and predicted sequences
np.save('../../../models/TGDS/' + str(MODEL_ID) + '/predictions', yhat_train_unscaled)
fig.savefig('../../../reports/figures/theory_guided_charge-' + str(MODEL_ID) + '-validation&test_profiles.png')


# ### Prepare Data for Use Cases

# In[9]:


# load test sets dictionary
TEST_SETS =  np.load('../../../models/training_setup/test_sets.npy', allow_pickle=True)
TEST_SETS = TEST_SETS.item()

# select needed test profiles
test_profiles_usecase_1 = TEST_SETS['Reproduction']
test_profiles_usecase_2 = TEST_SETS['Abstraction']
test_profiles_usecase_3 = TEST_SETS['Generalization']

# prepare input data
X_case_1, y_case_1, _ = util.prepare_feature_engineering_input(HYPER_PARAMS, test_profiles_usecase_1, HYPER_PARAMS['stack'], HYPER_PARAMS['cell'])
X_case_2, y_case_2, _ = util.prepare_feature_engineering_input(HYPER_PARAMS, test_profiles_usecase_2, HYPER_PARAMS['stack'], HYPER_PARAMS['cell'])
X_case_3, y_case_3, _ = util.prepare_feature_engineering_input(HYPER_PARAMS, test_profiles_usecase_3, HYPER_PARAMS['stack'], HYPER_PARAMS['cell'])


# ### Test Model on Use Cases

# In[10]:


print('Use Case 1:', test_profiles_usecase_1)
print('Use Case 2:', test_profiles_usecase_2)
print('Use Case 3:', test_profiles_usecase_3)

train_mse, case_1_mse, case_2_mse, case_3_mse, fig = lstm.test_usecases(X_train, y_train, X_case_1, y_case_1, X_case_2, y_case_2, X_case_3, y_case_3, scalers_train)


# In[11]:


fig.savefig('../../../reports/figures/data_baseline-' + str(MODEL_ID) + '-use_cases.png')
columns = ["Name", "Timestamp", "Model ID", "Train MSE", "Case 1 MSE", "Case 2 MSE", "Case 3 MSE"]
df = pd.DataFrame(columns=columns)

timestamp = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")

new_df = pd.DataFrame([["Initialization", timestamp, MODEL_ID, train_mse, case_1_mse, case_2_mse, case_3_mse]], columns=columns)

df = pd.concat(([df, new_df]))
df.to_csv('../../../reports/results/experiments.csv', mode='a', float_format='%.10f', index=False, header=False)


# In[12]:


device = cuda.get_current_device()
device.reset()

