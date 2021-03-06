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
# index = 0

# def residual_loss(params, u_pred):
#     """Wrapper function for residual loss.

#     Args:
#         params (dict): 
#             A dictionary containing the hyperparameters
#         u_pred (numpy.ndarray): 
#             The predictions made by the theory-based model
#     Returns: 
#         The residual loss
#     """
#     # wrapper function needed for the custom loss function to be accepted from keras
#     def loss(y_true, y_pred):
#         global index

#         delta = abs(y_true - u_pred[index])
# #         print(delta)
#         index += 1
        
#         # reset index
#         if (index == u_pred.shape[0]):
#             index = 0

#         return losses.mean_squared_error(y_pred, delta)
#     return loss

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
        model.compile(run_eagerly=True, optimizer=params['optimizer'], loss='mse', metrics=['mse', params['metric']])
        
        # save model parameters
        self.model = model
        self.params = params
        return None
        
    def train(self, params, X, y, scalers_train):
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
        mse = history.history['mse']
        mae = history.history['mae']
        epochs = range(1,len(loss)+1)
        
        print('Training time:', str(round(np.sum(time_callback.times), 3)) +'s')
        
        fig, _ = plt.subplots(figsize=(8,5))
        plt.plot(epochs, loss,'-o', color='green', label='training loss')
        plt.plot(epochs, mse,'-o', color='blue', label='mean squared error')
        plt.plot(epochs, mae,'-o', color='red',label='mean absolute error')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.show()
        
        # save parameters
        self.history = history
        self.scalers_train = scalers_train
        return time_callback, fig
    
    def test(self, X_train, y_train, u_train, X_validation, y_validation, u_validation, X_test, y_test, u_test, scalers_train):
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
        # --------- predict on training ---------
        time_train = TimeHistory()
        yhat_train = self.model.predict(X_train, callbacks=[time_train], verbose=1)
        # combine gap prediction with theory-based output
        yhat_plus_uhat_train = yhat_train + u_train
        y_true_train = y_train + u_train 
        # rescale output
        yhat_plus_uhat_train_unscaled = scalers_train.inverse_transform(yhat_plus_uhat_train)
        y_true_train_unscaled = scalers_train.inverse_transform(y_true_train)
        u_train_unscaled = scalers_train.inverse_transform(u_train)
        print('Prediction time on Training Set: ', str(round(np.sum(time_train.times), 3)) + 's')

        # --------- predict on validation ---------
        time_val = TimeHistory()
        yhat_val = self.model.predict(X_validation, callbacks=[time_val], verbose=1)
        # combine gap prediction with theory-based output
        yhat_plus_uhat_val = yhat_val + u_validation
        y_true_val = y_validation + u_validation 
        # rescale output
        yhat_plus_uhat_val_unscaled = scalers_train.inverse_transform(yhat_plus_uhat_val)
        y_true_val_unscaled = scalers_train.inverse_transform(y_true_val)
        u_val_unscaled = scalers_train.inverse_transform(u_validation)
        print('Prediction time on Validation Set: ', str(round(np.sum(time_val.times), 3)) + 's')
        
        # --------- predict on test ---------
        time_test = TimeHistory()
        yhat_test = self.model.predict(X_test, callbacks=[time_test], verbose=1)
        # combine gap prediction with theory-based output
        yhat_plus_uhat_test = yhat_test + u_test
        y_true_test = y_test + u_test 
        # rescale output
        yhat_plus_uhat_test_unscaled = scalers_train.inverse_transform(yhat_plus_uhat_test)
        y_true_test_unscaled = scalers_train.inverse_transform(y_true_test)
        u_test_unscaled = scalers_train.inverse_transform(u_test)
        print('Prediction time on Test Set: ', str(round(np.sum(time_test.times), 3)) + 's')
        
        # --------- compute error ---------
        train_mse = metrics.mean_squared_error(y_true_train_unscaled, yhat_plus_uhat_train_unscaled)
        validation_mse = metrics.mean_squared_error(y_true_val_unscaled, yhat_plus_uhat_val_unscaled)
        test_mse = metrics.mean_squared_error(y_true_test_unscaled, yhat_plus_uhat_test_unscaled)
        
        train_mae = metrics.mean_absolute_error(y_true_train_unscaled, yhat_plus_uhat_train_unscaled)
        validation_mae = metrics.mean_absolute_error(y_true_val_unscaled, yhat_plus_uhat_val_unscaled)
        test_mae = metrics.mean_absolute_error(y_true_test_unscaled, yhat_plus_uhat_test_unscaled)
        
        train_max = metrics.max_error(y_true_train_unscaled, yhat_plus_uhat_train_unscaled)
        validation_max = metrics.max_error(y_true_val_unscaled, yhat_plus_uhat_val_unscaled)
        test_max = metrics.max_error(y_true_test_unscaled, yhat_plus_uhat_test_unscaled)
        
        # --------- visualize results ---------
        print('###########################################################')
        error_table = tabulate([['MSE (\u03BCV)', round(train_mse, 7) * 1000000, round(validation_mse, 7) * 1000000, round(test_mse, 7) * 1000000], 
          ['MAE (V)', round(train_mae, 4), round(validation_mae, 4), round(test_mae, 4)], 
          ['MaxE (V)', round(train_max, 4), round(validation_max, 4), round(test_max, 4)]], headers=['Training', 'Validation', 'Test'])
        print(error_table)
        print('###########################################################')

        fig,_ = plt.subplots(figsize=(14,10))
        plt.subplot(2,2,1)
        time_val = np.arange(yhat_plus_uhat_val_unscaled.shape[0]) * 0.25
        plt.plot(time_val, yhat_plus_uhat_val_unscaled, color='blue', label='residual + theory')
        plt.plot(time_val, y_true_val_unscaled, color='green', label='true')
        plt.plot(time_val, u_val_unscaled, color='red', label='theory')
        plt.ylabel('voltage (V)')
        plt.title('Validation Data')
        plt.legend()
        plt.subplot(2,2,3)
        delta_val = np.abs(yhat_plus_uhat_val_unscaled - y_true_val_unscaled)
        plt.plot(time_val, delta_val, color='m', label='predicted - measured')
        plt.ylabel('voltage (V)')
        plt.xlabel('time (s)')
        plt.title('Absolute Error')
        plt.legend()
        plt.subplot(2,2,2)
        time_test = np.arange(yhat_plus_uhat_test_unscaled.shape[0]) * 0.25
        plt.plot(time_test, yhat_plus_uhat_test_unscaled, color='blue', label='residual + theory')
        plt.plot(time_test, y_true_test_unscaled, color='green', label='true')
        plt.plot(time_test, u_test_unscaled, color='red', label='theory')
        plt.ylabel('voltage (V)')
        plt.title('Test Data')
        plt.legend()
        plt.subplot(2,2,4)
        delta_test = np.abs(yhat_plus_uhat_test_unscaled - y_true_test_unscaled)
        plt.plot(time_test, delta_test, color='m', label='predicted - measured')
        plt.ylabel('voltage (V)')
        plt.xlabel('time (s)')
        plt.title('Absolute Error')
        plt.legend()
        plt.show()
        
        return yhat_plus_uhat_val_unscaled, fig
   
    def test_usecases(self, X_train, y_train, u_train, X_case_1, y_case_1, u_case_1, X_case_2, y_case_2, u_case_2, X_case_3, y_case_3, u_case_3, scalers_train):
        # --------- predict on training ---------
        time_train = TimeHistory()
        yhat_train = self.model.predict(X_train, callbacks=[time_train], verbose=1)
        # combine gap prediction with theory-based output
        yhat_plus_uhat_train = yhat_train + u_train
        y_true_train = y_train + u_train 
        # rescale output
        yhat_plus_uhat_train_unscaled = scalers_train.inverse_transform(yhat_plus_uhat_train)
        y_true_train_unscaled = scalers_train.inverse_transform(y_true_train)
        u_train_unscaled = scalers_train.inverse_transform(u_train)
        print('Prediction time on Training Set: ', str(round(np.sum(time_train.times), 3)) + 's')

        # --------- predict on use case 1 ---------
        time_case_1 = TimeHistory()
        yhat_case_1 = self.model.predict(X_case_1, callbacks=[time_case_1], verbose=1)
        # combine gap prediction with theory-based output
        yhat_plus_uhat_case_1 = yhat_case_1 + u_case_1
        y_true_case_1 = y_case_1 + u_case_1
        # rescale output
        yhat_plus_uhat_case_1_unscaled = scalers_train.inverse_transform(yhat_plus_uhat_case_1)
        y_true_case_1_unscaled = scalers_train.inverse_transform(y_true_case_1)
        u_case_1_unscaled = scalers_train.inverse_transform(u_case_1)
        print('Prediction time on Use Case 1: ', str(round(np.sum(time_case_1.times), 3)) + 's')
        
        # --------- predict on use case 2 ---------
        time_case_2 = TimeHistory()
        yhat_case_2 = self.model.predict(X_case_2, callbacks=[time_case_2], verbose=1)
        # combine gap prediction with theory-based output
        yhat_plus_uhat_case_2 = yhat_case_2 + u_case_2
        y_true_case_2 = y_case_2 + u_case_2
        # rescale output
        yhat_plus_uhat_case_2_unscaled = scalers_train.inverse_transform(yhat_plus_uhat_case_2)
        y_true_case_2_unscaled = scalers_train.inverse_transform(y_true_case_2)
        u_case_2_unscaled = scalers_train.inverse_transform(u_case_2)
        print('Prediction time on Use Case 2: ', str(round(np.sum(time_case_2.times), 3)) + 's')
        
        # --------- predict on use case 3 ---------
        time_case_3 = TimeHistory()
        yhat_case_3 = self.model.predict(X_case_3, callbacks=[time_case_3], verbose=1)
        # combine gap prediction with theory-based output
        yhat_plus_uhat_case_3 = yhat_case_3 + u_case_3
        y_true_case_3 = y_case_3 + u_case_3
        # rescale output
        yhat_plus_uhat_case_3_unscaled = scalers_train.inverse_transform(yhat_plus_uhat_case_3)
        y_true_case_3_unscaled = scalers_train.inverse_transform(y_true_case_3)
        u_case_3_unscaled = scalers_train.inverse_transform(u_case_3)
        print('Prediction time on Use Case 3: ', str(round(np.sum(time_case_3.times), 3)) + 's')
        
        # --------- compute error ---------
        train_mse = metrics.mean_squared_error(y_true_train_unscaled, yhat_plus_uhat_train_unscaled)
        case_1_mse = metrics.mean_squared_error(y_true_case_1_unscaled, yhat_plus_uhat_case_1_unscaled)
        case_2_mse = metrics.mean_squared_error(y_true_case_2_unscaled, yhat_plus_uhat_case_2_unscaled)
        case_3_mse = metrics.mean_squared_error(y_true_case_3_unscaled, yhat_plus_uhat_case_3_unscaled)

        train_mae = metrics.mean_absolute_error(y_true_train_unscaled, yhat_plus_uhat_train_unscaled)
        case_1_mae = metrics.mean_absolute_error(y_true_case_1_unscaled, yhat_plus_uhat_case_1_unscaled)
        case_2_mae = metrics.mean_absolute_error(y_true_case_2_unscaled, yhat_plus_uhat_case_2_unscaled)
        case_3_mae = metrics.mean_absolute_error(y_true_case_3_unscaled, yhat_plus_uhat_case_3_unscaled)
        
        train_max = metrics.max_error(y_true_train_unscaled, yhat_plus_uhat_train_unscaled)
        case_1_max = metrics.max_error(y_true_case_1_unscaled, yhat_plus_uhat_case_1_unscaled)
        case_2_max = metrics.max_error(y_true_case_2_unscaled, yhat_plus_uhat_case_2_unscaled)
        case_3_max = metrics.max_error(y_true_case_3_unscaled , yhat_plus_uhat_case_3_unscaled)
        
        # --------- visualize results ---------
        print('##############################################################')
        error_table = tabulate([['MSE (\u03BCV)', round(train_mse, 7) * 1000000, round(case_1_mse, 7) * 1000000, round(case_2_mse, 7) * 1000000, round(case_3_mse, 7) * 1000000], 
          ['MAE (V)', round(train_mae, 4), round(case_1_mae, 4), round(case_2_mae, 4), round(case_3_mae, 4)], 
          ['MaxE (V)', round(train_max, 4), round(case_1_max, 4), round(case_2_max, 4), round(case_3_max, 4)]], headers=['Training', 'Use Case 1', 'Use Case 2', 'Use Case 3'])
        print(error_table)
        print('##############################################################')
        
        fig,_ = plt.subplots(figsize=(7,16))
        plt.subplot(3,1,1)
        time_case_1 = np.arange(yhat_plus_uhat_case_1_unscaled.shape[0]) * 0.25
        plt.plot(time_case_1, yhat_plus_uhat_case_1_unscaled, color='blue', label='residual + theory')
        plt.plot(time_case_1, y_true_case_1_unscaled, color='green', label='true')
        plt.plot(time_case_1, u_case_1_unscaled, color='red', label='theory')
        plt.ylabel('voltage (V)')
        plt.title('Use Case 1')
        plt.legend()
        plt.subplot(3,1,2)
        time_case_2 = np.arange(yhat_plus_uhat_case_2_unscaled.shape[0]) * 0.25
        plt.plot(time_case_2, yhat_plus_uhat_case_2_unscaled, color='blue', label='residual + theory')
        plt.plot(time_case_2, y_true_case_2_unscaled, color='green', label='true')
        plt.plot(time_case_2, u_case_2_unscaled, color='red', label='theory')
        plt.ylabel('voltage (V)')
        plt.title('Use Case 2')
        plt.legend()
        plt.subplot(3,1,3)
        time_case_3 = np.arange(yhat_plus_uhat_case_3_unscaled.shape[0]) * 0.25
        plt.plot(time_case_3, yhat_plus_uhat_case_3_unscaled, color='blue', label='residual + theory')
        plt.plot(time_case_3, y_true_case_3_unscaled, color='green', label='true')
        plt.plot(time_case_3, u_case_3_unscaled, color='red', label='theory')
        plt.ylabel('voltage (V)')
        plt.ylabel('time (s)')
        plt.title('Use Case 3')
        plt.legend()
        plt.show()

        
        return fig
        