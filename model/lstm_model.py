"""
Module containing a class modelling an LSTM network for voltage time series prediciton.
"""
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import sklearn.metrics as metrics
import data_preprocessing as util

from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import losses
from tensorflow.keras import callbacks
from tabulate import tabulate

import tensorflow.python.util.deprecation as deprecation
# used to hide deprecation warning raised by tensorflow
deprecation._PRINT_DEPRECATION_WARNINGS = False 

# ---------------------------------------------------- Callbacks ----------------------------------------------------
class TimeHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        
    def on_predict_begin(self, logs={}):
        self.times = []
        self.epoch_time_start = time.time()

    def on_predict_end(self, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# ---------------------------------------------------- Custom Loss Functions ----------------------------------------------------

def approximation_loss(y_true, y_pred):
    """Computes the approximation loss as described in [1].

    This loss term penalizes predictions which lay outside of the defined value range of the output.
    The respective value range has to be set by the keys 'y_l' (lower end) and 'y_u' (upper end) in the params dictionary.

    [1] http://people.cs.vt.edu/ramakris/papers/PID5657885.pdf

    Args:
        y_pred (numpy.ndarray):
            An array of the predicted output values
        params (dict): 
            A dictionary containing the hyperparameters
    Returns:
        The approximation loss of the given predictions. Zero if the predictions follow the approximation constraint, positive if not.
    """
    y_lower = backend.min(y_true)
    y_upper = backend.max(y_true)
    loss = backend.sum(backend.relu(y_lower - y_pred) + backend.relu(y_pred - y_upper))
    
    # to visualize loss during training
    if (loss.numpy() != 0,0):
        print(' - apx:', loss.numpy())
    return loss


def monotonicity_loss(y_true, y_pred):
    """Computes the monotonicity loss as described in [1].
    
    This loss term penalizes predictions which are not monotonicaly accending or decending according to the behaviour of the ground truth.

    [1] http://people.cs.vt.edu/ramakris/papers/PID5657885.pdf

    Args:
        y_true (numpy.ndarray): 
            An array of the groundtruth output values
        y_pred (numpy.ndarray): 
            An array of the predicted output values
    Returns:
        The monotonicity loss of the given predictions. Zero if the predictions follow the monotonicity constraint, positive if not.
    """
    loss = 0
    loss_changed = False

    for i in range(len(y_true) - 1):
        if y_true[i] < y_true[i+1] and y_pred[i] >= y_pred[i+1]:
            # monotonic accent 
            loss += backend.abs(y_pred[i] - y_pred[i+1])
            loss_changed = True
            
        elif y_true[i] > y_true[i+1] and y_pred[i] <= y_pred[i+1]:
            # monotonic decent
            loss += backend.abs(y_pred[i+1] - y_pred[i])
            loss_changed = True
  
    if (loss_changed):
        print(' - mon:', loss.numpy()[0])

    return loss


def combine_losses(params):
    """Combines multiple custom loss functions.

    The loss functions provided by the 'loss_funcs' key in the params array will be combined in a weighted sum. 
    Weights are determined by the respective 'lambda_xyz' key. 

    Args:
        params (dict): 
            A dictionary containing the hyperparameters
    Returns: 
        The combined loss value.
    """
    # wrapper function needed for the custom loss function to be accepted from keras
    def loss(y_true, y_pred):
        accumulated_loss = 0
        loss_functions = params['loss_funcs']

        # !! add custom loss function here !!
        if 'mse' in loss_functions:
            accumulated_loss += params['lambda_mse'] * losses.mean_squared_error(y_true, y_pred)
        if 'apx' in loss_functions:
            accumulated_loss += params['lambda_apx'] * approximation_loss(y_true, y_pred)
        if 'mon' in loss_functions:
            accumulated_loss += params['lambda_mon'] * monotonicity_loss(y_true, y_pred)

        return accumulated_loss
    return loss

# ---------------------------------------------------- LSTM Model  ----------------------------------------------------

class Model: 
    """Responsible for managing the neural network architecture which is used to predict voltage time series data.

    Model is suited to work with the FOBSS data set (http://dbis.ipd.kit.edu/download/FOBSS_final.pdf) but can also be used with
    other kinds of current and voltage data.

    Attributes:
        model (tensorflow.python.keras.engine.sequential.Sequential): 
            A keras object representing the compiled model
            
        params (dict): 
            A dictionary containing the hyperparameters
            
        history (tensorflow.python.keras.callbacks.History): 
            A report of the training procedure
    """
   
    def initialize(self, params):
        """Initializes the LSTM model.

        For visualization purposes, a summary of the model will be printed.

        Args:
            params (dict): 
                A dictionary containing the hyperparameters
        """
        # --------- create model ---------
        model = tf.keras.Sequential()
        # layer 1
        model.add(layers.LSTM(units=params['n_lstm_units_1'], input_shape=(params['n_steps'], params['n_features']), return_sequences=True))
        model.add(layers.LeakyReLU(alpha=params['alpha_1']))
        # layer 2
        model.add(layers.LSTM(units=params['n_lstm_units_2']))
        model.add(layers.LeakyReLU(alpha=params['alpha_1']))
        # output layer
        model.add(layers.Dense(1, activation=params['activation_output_layer']))
        model.summary()
        
        # --------- compile model ---------
        custom_loss = combine_losses(params)
        
        model.compile(run_eagerly=True, optimizer=params['optimizer'], loss=custom_loss, metrics=['mse', params['metric']])
        
        # save model parameters
        self.model = model
        self.params = params
        return None
    
    def train(self, X, y, scalers):
        """Trains the LSTM model.

        For visualization purposes, the MSE and MAE over all training epochs will be ploted.

        Args:
            X (numpy.ndarray): 
                The input data
                
            y (numpy.ndarray): 
                The groundtruth output data
            
            scalers (tuple):
                The scaler objects which were used to scale X and y
        Returns:
            The time_callback which is used to measure the time needed to train the model. In adition the matplotlib 
            figure used to plot the visualization. This is needed so that the plots can be saved at the appropriate location.
        """
        # --------- train model ---------
        time_callback = TimeHistory()
        history = self.model.fit(X, y, epochs=self.params['n_epochs'], callbacks=[time_callback], verbose=1)
        
        # --------- visualize results ---------
        loss = history.history['loss']
        mse = history.history['mse']
        mae = history.history['mae']
        epochs = range(1,len(loss)+1)
        
        print('Training time:', str(round(np.sum(time_callback.times), 3)) +'s')
        
        fig, _ = plt.subplots(figsize=(8,5))
        plt.plot(epochs, loss,'-o', color='green', label='training loss')
        plt.plot(epochs, mse,'-o', color='blue', label='mean squared error')
        plt.plot(epochs, mae,'-o', color='red',label='mean absolute error')
        plt.legend()
        plt.show()
        
        # save parameters
        self.history = history
        self.scalers_train = scalers
        return time_callback, fig


    def test(self, X_train, y_train, X_validation, y_validation, X_test, y_test, scalers):
        """Tests the LSTM model on validation and test data.
    
        For visualization purposes, a table with several metrics for training, validation and test data 
        will be printed in adition to plots of the validation and test profiles used.
 
        Args:
            X_train (numpy.ndarray):
                The training input data used in Model.train()
                
            y_train (numpy.ndarray):
                The training output data used in Model.train()
            
            X_validation (numpy.ndarray): 
                validation input data
                
            y_validation (numpy.ndarray): 
                validation output data
                
            X_test (numpy.ndarray): 
                test input data
                
            y_test (numpy.ndarray): 
                test output data

            scalers (tuple):
                The scaler objects which were used to scale X and y in training, validation and test data
                
        Returns:
            A Tuple containing the predicted train, validation and test profiles. In adition the matplotlib figure 
            used to plot the visualization is returned. This is needed so that the plots can be saved at the appropriate location.
        """
        # --------- predict on data ---------
        time_train = TimeHistory()
        yhat_train = self.model.predict(X_train, callbacks=[time_train], verbose=1)
        yhat_train_unscaled = scalers[0][1].inverse_transform(yhat_train)
        y_train_unscaled = scalers[0][1].inverse_transform(y_train)
        print('Prediction time on Training Set: ', str(round(np.sum(time_train.times), 3)) + 's')
        
        time_val = TimeHistory()
        yhat_validation = self.model.predict(X_validation, callbacks=[time_val], verbose=1)
        yhat_validation_unscaled = scalers[1][1].inverse_transform(yhat_validation)
        y_validation_unscaled = scalers[1][1].inverse_transform(y_validation)
        print('Prediction time on Validation Set: ', str(round(np.sum(time_val.times), 3)) + 's')
        
        time_test = TimeHistory()
        yhat_test = self.model.predict(X_test, callbacks=[time_test], verbose = 1)
        yhat_test_unscaled = scalers[2][1].inverse_transform(yhat_test)
        y_test_unscaled = scalers[2][1].inverse_transform(y_test)
        print('Prediction time on Test Set: ', str(round(np.sum(time_test.times), 3)) + 's')

        # --------- compute error ---------
        train_mse = metrics.mean_squared_error(y_train_unscaled, yhat_train_unscaled)
        validation_mse = metrics.mean_squared_error(y_validation_unscaled, yhat_validation_unscaled)
        test_mse = metrics.mean_squared_error(y_test_unscaled, yhat_test_unscaled)
        
        train_mae = metrics.mean_absolute_error(y_train_unscaled, yhat_train_unscaled)
        validation_mae = metrics.mean_absolute_error(y_validation_unscaled, yhat_validation_unscaled)
        test_mae = metrics.mean_absolute_error(y_test_unscaled, yhat_test_unscaled)
        
        train_max = metrics.max_error(y_train_unscaled, yhat_train_unscaled)
        validation_max = metrics.max_error(y_validation_unscaled, yhat_validation_unscaled)
        test_max = metrics.max_error(y_test_unscaled, yhat_test_unscaled)
        
        # --------- visualize results ---------
        print('###########################################################')
        error_table = tabulate([['MSE (\u03BCV)', round(train_mse, 7) * 1000000, round(validation_mse, 7) * 1000000, round(test_mse, 7) * 1000000], 
          ['MAE (V)', round(train_mae, 4), round(validation_mae, 4), round(test_mae, 4)], 
          ['MaxE (V)', round(train_max, 4), round(validation_max, 4), round(test_max, 4)]], headers=['Training', 'Validation', 'Test'])
        print(error_table)
        print('###########################################################')
        
        time = np.arange(yhat_test_unscaled.shape[0]) * 0.25
        delta = np.abs(yhat_test_unscaled - y_test_unscaled)
        
        fig,_ = plt.subplots(figsize=(7,10))
#         plt.subplot(2,1,1)
#         time = np.arange(yhat_validation_unscaled.shape[0]) * 0.25
#         plt.plot(time, yhat_validation_unscaled, color='red', label='predicted')
#         plt.plot(time, y_validation_unscaled, color='blue', label='measured')
#         plt.title('Validation Data')
#         plt.legend()
        plt.subplot(2,1,1)
        plt.plot(time, yhat_test_unscaled, color='red', label='predicted')
        plt.plot(time, y_test_unscaled, color='blue', label='measured')
        plt.ylabel('voltage (V)')
        plt.title('Test Data')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(time, delta, color='m', label='predicted - measured')
        plt.ylabel('voltage (V)')
        plt.xlabel('time (s)')
        plt.title('Absolute Error')
        plt.legend()
        plt.show()
        
        return yhat_train_unscaled, yhat_validation_unscaled, yhat_test_unscaled, delta, time, fig


    def test_usecases(self, X_train, y_train, X_test_1, y_test_1, X_test_2, y_test_2, X_test_3, y_test_3, scalers):
        # --------- predict on data ---------
        time_train = TimeHistory()
        yhat_train = self.model.predict(X_train, callbacks=[time_train], verbose=1)
        yhat_train_unscaled = scalers[0][1].inverse_transform(yhat_train)
        y_train_unscaled = scalers[0][1].inverse_transform(y_train)
        print('Prediction time on Training Set: ', str(round(np.sum(time_train.times), 3)) + 's')
        
        time_test_1 = TimeHistory()
        yhat_test_1 = self.model.predict(X_test_1, callbacks=[time_test_1], verbose=1)
        yhat_test_1_unscaled = scalers[1][1].inverse_transform(yhat_test_1)
        y_test_1_unscaled = scalers[1][1].inverse_transform(y_test_1)
        print('Prediction time on Use Case 1: ', str(round(np.sum(time_test_1.times), 3)) + 's')
        
        time_test_2 = TimeHistory()
        yhat_test_2 = self.model.predict(X_test_2, callbacks=[time_test_2], verbose=1)
        yhat_test_2_unscaled = scalers[2][1].inverse_transform(yhat_test_2)
        y_test_2_unscaled = scalers[2][1].inverse_transform(y_test_2)
        print('Prediction time on Use Case 2: ', str(round(np.sum(time_test_2.times), 3)) + 's')

        time_test_3 = TimeHistory()
        yhat_test_3 = self.model.predict(X_test_3, callbacks=[time_test_3], verbose=1)
        yhat_test_3_unscaled = scalers[3][1].inverse_transform(yhat_test_3)
        y_test_3_unscaled = scalers[3][1].inverse_transform(y_test_3)
        print('Prediction time on Use Case 3: ', str(round(np.sum(time_test_3.times), 3)) + 's')

        # --------- compute error ---------
        train_mse = metrics.mean_squared_error(y_train_unscaled, yhat_train_unscaled)
        test_1_mse = metrics.mean_squared_error(y_test_1_unscaled, yhat_test_1_unscaled)
        test_2_mse = metrics.mean_squared_error(y_test_2_unscaled, yhat_test_2_unscaled)
        test_3_mse = metrics.mean_squared_error(y_test_3_unscaled, yhat_test_3_unscaled)

        train_mae = metrics.mean_absolute_error(y_train_unscaled, yhat_train_unscaled)
        test_1_mae = metrics.mean_absolute_error(y_test_1_unscaled, yhat_test_1_unscaled)
        test_2_mae = metrics.mean_absolute_error(y_test_2_unscaled, yhat_test_2_unscaled)
        test_3_mae = metrics.mean_absolute_error(y_test_3_unscaled, yhat_test_3_unscaled)
        
        train_max = metrics.max_error(y_train_unscaled, yhat_train_unscaled)
        test_1_max = metrics.max_error(y_test_1_unscaled, yhat_test_1_unscaled)
        test_2_max = metrics.max_error(y_test_2_unscaled, yhat_test_2_unscaled)
        test_3_max = metrics.max_error(y_test_3_unscaled, yhat_test_3_unscaled)
        
        # --------- visualize results ---------
        print('##############################################################')
        error_table = tabulate([['MSE (\u03BCV)', round(train_mse, 7) * 1000000, round(test_1_mse, 7) * 1000000, round(test_2_mse, 7) * 1000000, round(test_3_mse, 7) * 1000000], 
          ['MAE (V)', round(train_mae, 4), round(test_1_mae, 4), round(test_2_mae, 4), round(test_3_mae, 4)], 
          ['MaxE (V)', round(train_max, 4), round(test_1_max, 4), round(test_2_max, 4), round(test_3_max, 4)]], headers=['Training', 'Use Case 1', 'Use Case 2', 'Use Case 3'])
        print(error_table)
        print('##############################################################')
    
# ---------------------------------------------------- Residual LSTM Model ----------------------------------------------------
# needed for residual loss computation
    lower_index_res = 0
    upper_index_res = 0
def residual_loss(params, u_pred):
    """Wrapper function for residual loss.

    Args:
        params (dict): 
            A dictionary containing the hyperparameters
        u_pred (numpy.ndarray): 
            The predictions made by the theory-based model
    Returns: 
        The residual loss
    """
    # wrapper function needed for the custom loss function to be accepted from keras
    def loss(y_true, y_pred):
        global lower_index_res
        global upper_index_res
        lower_index_res = upper_index_res
        upper_index_res += y_true.shape[0]

        u_pred_sliced = u_pred[lower_index_res:upper_index_res]
        delta = y_true - u_pred_sliced

        return losses.mean_squared_error(y_pred, delta)
    return loss

class Residual_Model: 
    """Responsible for managing the neural network architecture which is used for residual learning. 
    The goal is to predict the gap between the theory-based model and the ground-truth, not the voltage itself.

    Residual_Model is suited to work with the FOBSS data set (http://dbis.ipd.kit.edu/download/FOBSS_final.pdf) but can also be used with
    other kinds of current and voltage data.

    Attributes:
        model (tensorflow.python.keras.engine.sequential.Sequential): 
            A keras object representing the compiled model
            
        params (dict): 
            A dictionary containing the hyperparameters
            
        history (tensorflow.python.keras.callbacks.History): 
            A report of the training procedure
    """
      
    def initialize(self, params):
        """Initializes the residual LSTM model.

        For visualization purposes, a summary of the model will be printed.

        Args:
            params (dict): 
                A dictionary containing the hyperparameters
        """
        
        # --------- create model ---------
        model = tf.keras.Sequential()
        # layer 1
        model.add(layers.LSTM(units=params['n_lstm_units_1'], input_shape=(params['n_steps'], params['n_features']), return_sequences=True))
        model.add(layers.LeakyReLU(alpha=params['alpha_1']))
        # layer 2
        model.add(layers.LSTM(units=params['n_lstm_units_2']))
        model.add(layers.LeakyReLU(alpha=params['alpha_1']))
        # output layer
        model.add(layers.Dense(1, activation=params['activation']))
        model.summary()
        
        # --------- compile model ---------
        # load residual data for specific profile
        u_pred = np.load('trained_models/TGDS/9079/predictions.npy') # TODO: change this to the appropriate EC-Model path
        u_pred = np.repeat(u_pred, params['n_epochs'])
        u_pred_preprocessed, scaler_res = util.preprocess_raw_data(params, u_pred)

        custom_loss = residual_loss(params, u_pred_preprocessed)
        
        model.compile(run_eagerly=True, optimizer=params['optimizer'], loss=custom_loss)
        
        # save model parameters
        self.model = model
        self.params = params
        return None
    
    def train(self, X, y, scalers):
        """Trains the residual LSTM model.

        For visualization purposes, the training error over all epochs will be ploted.

        Args:
            X (numpy.ndarray): 
                The input data
                
            y (numpy.ndarray): 
                The groundtruth output data
            
            scalers (tuple):
                The scaler objects which were used to scale X and y
        Returns:
            The time_callback which is used to measure the time needed to train the model. In adition the matplotlib 
            figure used to plot the visualization. This is needed so that the plots can be saved at the appropriate location.
        """
        
        # --------- train model ---------
        time_callback = TimeHistory()
        history = self.model.fit(X, y, epochs=self.params['n_epochs'], callbacks=[time_callback], verbose=1)
        
        # --------- visualize results ---------
        loss = history.history['loss'] 
        epochs = range(1,len(loss)+1)
        
        print('Training time:', np.sum(time_callback.times))
        
        fig, _ = plt.subplots(figsize=(8,5))
        plt.plot(epochs, loss,'-o', color='green', label='training loss')
        plt.legend()
        plt.show()
        
        # save parameters
        self.history = history
        self.scalers_train = scalers
        return time_callback, fig
    
    def test(self, X_train, y_train, scalers):
        """Tests the residual LSTM model on test data.

        For visualization purposes, a table with several metrics for the training data 
        will be printed in adition to plots of the used profile.

        Args:
            X_train (numpy.ndarray):
                The training input data used in Model.train()

            y_train (numpy.ndarray):
                The training output data used in Model.train()

            scalers (tuple):
                The scaler objects which were used to scale X and y in training and test data

        Returns:
            A Tuple containing the error of the prediction on training data. In adition the matplotlib figure 
            used to plot the visualization is returned. This is needed so that the plots can be saved at the appropriate location.
        """
        # load residual data for specific profile
        u_pred = np.load('trained_models/TGDS/9079/predictions.npy') # TODO: change this to the appropriate EC-Model path
        plt.plot(u_pred)

        # --------- predict on data ---------
        time_callback = TimeHistory()
        yhat_train = self.model.predict(X_train, callbacks=[time_callback], verbose=1)
        yhat_train_unscaled = scalers[0][1].inverse_transform(yhat_train)
        y_train_unscaled = scalers[0][1].inverse_transform(y_train)
        
        # combine gap prediction with theory-based output
        yhat = u_pred + yhat_train 

        # --------- compute error ---------
        train_mse = metrics.mean_squared_error(y_train_unscaled, yhat)
        train_mae = metrics.mean_absolute_error(y_train_unscaled, yhat)
        train_max = metrics.max_error(y_train_unscaled, yhat)

        # --------- visualize results ---------
        print('###########################################################')
        error_table = tabulate([['MSE', round(train_mse, 8)], 
          ['MAE', round(train_mae, 4)], 
          ['MaxE', round(train_max, 4)]], headers=['Training', 'Validation', 'Test'])
        print(error_table)
        print('###########################################################')
        
        print('Training time:', np.sum(time_callback.times))
        
        time = np.arange(yhat_train.shape[0]) * 0.25
        delta = np.abs(yhat_train_unscaled - y_train_unscaled)
        
        fig,_ = plt.subplots(figsize=(7,10))
        plt.subplot(1,2,1)
        # plt.plot(time, yhat_train, color='red', label='predicted')
        plt.plot(time, yhat, color='red', label='predicted + theory')
        plt.plot(time, y_train, color='blue', label='measured')
        plt.plot(time, u_pred, color='green', label='theory')
        plt.title('Training Data')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(time, delta, color='m', label='predicted - measured')
        plt.title('Absolute Error')
        plt.legend()
        plt.show()
        
        return yhat_train_unscaled, fig

    