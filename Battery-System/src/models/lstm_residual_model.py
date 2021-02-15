"""
Module containing a class modelling an LSTM network for voltage time series prediciton.
"""
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import sklearn.metrics as metrics
import tensorflow.python.util.deprecation as deprecation
# used to hide deprecation warning raised by tensorflow
deprecation._PRINT_DEPRECATION_WARNINGS = False 

from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import losses
from tensorflow.keras import callbacks
from tabulate import tabulate
from ..data.data_preprocessing import preprocess_raw_data

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
        model.add(layers.Dense(1, activation=params['activation_output_layer']))
        model.summary()
        
        # --------- compile model ---------
        # load residual data for specific profile
        u_pred = np.load('../../../models/TGDS/9079/predictions.npy') # TODO: change this to the appropriate EC-Model path, push this to git
        u_pred = np.repeat(u_pred, params['n_epochs'])
        u_pred_preprocessed, scaler_res = preprocess_raw_data(params, u_pred)

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
        u_pred = np.load('../../../models/TGDS/9079/predictions.npy') # TODO: change this to the appropriate EC-Model path
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

    