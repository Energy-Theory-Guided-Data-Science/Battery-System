"""
Module containing a class modelling an LSTM network for voltage time series prediciton.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import losses
from tabulate import tabulate

import tensorflow.python.util.deprecation as deprecation
# used to hide deprecation warning raised by tensorflow
deprecation._PRINT_DEPRECATION_WARNINGS = False 


def approximation_loss(y_pred, params):
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
    y_lower = params['y_l']
    y_upper = params['y_u']
    return backend.sum(backend.relu(params['y_l'] - y_pred) + backend.relu(y_pred - params['y_u']))


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
        
    for i in range(len(y_true) - 1):
        if y_true[i] < y_true[i+1] and y_pred[i] > y_pred[i+1]:
            # monotonic accent 
            loss += backend.relu(y_pred[i] - y_pred[i+1])
        elif y_true[i] > y_true[i+1] and y_pred[i] < y_pred[i+1]:
            # monotonic decent
            loss += backend.relu(y_pred[i+1] - y_pred[i])
    
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
            accumulated_loss += params['lambda_apx'] * approximation_loss(y_pred, params)
        if 'mon' in loss_functions:
            accumulated_loss += params['lambda_mon'] * monotonicity_loss(y_true, y_pred)

        return accumulated_loss
    return loss


class Model: 
    """Responsible for managing the neural network architecture which is used to predict voltage time series data.

    Model is suited to work with the FOBSS data set (http://dbis.ipd.kit.edu/download/FOBSS_final.pdf) but can also be used with
    different kinds of current and voltage data.

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
        model.add(layers.Dense(1, activation=params['activation']))
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
        """
        
        # --------- train model ---------
        history = self.model.fit(X, y, epochs=self.params['n_epochs'], verbose=1)
        
        # --------- visualize results ---------
        loss = history.history['loss']
        mse = history.history['mse']
        mae = history.history['mae']
        epochs = range(1,len(loss)+1)
        
        plt.subplots(figsize=(8,5))
        plt.plot(epochs, loss,'-o', color='green', label='training loss')
        plt.plot(epochs, mse,'-o', color='blue', label='mean squared error')
        plt.plot(epochs, mae,'-o', color='red',label='mean absolute error')
        plt.legend()
        plt.show()
        
        # save parameters
        self.history = history
        self.scalers_train = scalers
        return None


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
            A Tuple containin training, validation and test error (MSE)
        """
        
        # --------- predict on data ---------
        yhat_train = self.model.predict(X_train, verbose=1)
        yhat_train_unscaled = scalers[0][1].inverse_transform(yhat_train)
        y_train_unscaled = scalers[0][1].inverse_transform(y_train)
        
        yhat_validation = self.model.predict(X_validation, verbose=1)
        yhat_validation_unscaled = scalers[1][1].inverse_transform(yhat_validation)
        y_validation_unscaled = scalers[1][1].inverse_transform(y_validation)
        
        yhat_test = self.model.predict(X_test, verbose = 1)
        yhat_test_unscaled = scalers[2][1].inverse_transform(yhat_test)
        y_test_unscaled = scalers[2][1].inverse_transform(y_test)

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
        error_table = tabulate([['MSE', round(train_mse, 6), round(validation_mse, 6), round(test_mse, 6)], 
          ['MAE', round(train_mae, 4), round(validation_mae, 4), round(test_mae, 4)], 
          ['MaxE', round(train_max, 4), round(validation_max, 4), round(test_max, 4)]], headers=['Training', 'Validation', 'Test'])
        print(error_table)
        print('###########################################################')

        plt.subplots(figsize=(7,10))
        plt.subplot(2,1,1)  
        plt.plot(yhat_validation_unscaled, color='red', label='predicted')
        plt.plot(y_validation_unscaled, color='blue', label='measured')
        plt.title('Validation Data')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(yhat_test_unscaled, color='red', label='predicted')
        plt.plot(y_test_unscaled, color='blue', label='measured')
        plt.title('Test Data')
        plt.legend()
        plt.show()
        
        return train_mse, validation_mse, test_mse

    