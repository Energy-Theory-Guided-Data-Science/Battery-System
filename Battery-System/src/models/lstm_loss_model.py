"""
Module containing a class modelling an LSTM network for voltage time series prediciton using intermediate variables.
"""
import time
import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import sklearn.metrics as metrics
from tabulate import tabulate
import tensorflow.python.util.deprecation as deprecation
# used to hide deprecation warning raised by tensorflow
deprecation._PRINT_DEPRECATION_WARNINGS = False 
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import losses
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers


# ---------------------------------------------------- Callbacks -------------------------------------------------
class TimeHistory(callbacks.Callback):
    """Callback handler for keras to monitor the training time.

    Attributes:
        times (list): 
            A list of time step values for each consecutive training epoch
            
        epoch_time_start (double): 
            Variable to store the start time of an epoch to later compute the epoch training time
    """
    def on_train_begin(self, logs={}):
        """Initialize times array at the beginning of training.
        Called every time a training starts.
        """
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        """Starts the timer at the begining of an epoch.
        Called every time a new training epoch starts.
        """
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        """Computes the training time needed in this epoch and adds it to the time array.
        Called every time a training epoch finishes.
        """
        self.times.append(time.time() - self.epoch_time_start)
        
    def on_predict_begin(self, logs={}):
        """Resets the times array and starts a new epoch to later compute the time needed for a prediction. 
        Called every time a prediction starts.
        """
        self.times = []
        self.epoch_time_start = time.time()

    def on_predict_end(self, logs={}):
        """Computes the time needed to make a prediction.
        Called every time a prediction finishes.
        """
        self.times.append(time.time() - self.epoch_time_start)



# --------------------------------------------- LSTM Model  ------------------------------------------------------
class Model: 
    """Responsible for managing the neural network architecture using intermediate variables 
    and custom loss functions for training. 

    Attributes:
        model (tensorflow.python.keras.engine.sequential.Sequential): 
            A keras object representing the compiled model
            
        params (dict): 
            A dictionary containing the hyperparameters
            
        history (tensorflow.python.keras.callbacks.History): 
            A report of the training procedure
        
        scalers_train ((sklearn.preprocessing.MinMaxScaler, sklearn.preprocessing.MinMaxScaler)):
            A tuple of scaler objects used to scale and rescale X and y for training
    """
    
    #  ocv_inverse(v) ----
    def ocv_inverse_exact(self, v):
        """Returns the exact SOC using the expert-provided OCV curve. 

        Args:
            v (double):
                The OCV of the required SOC

        Returns:
            The required SOC. 
        """
        ocv = self.ocv_curve[:,2]
        delta = np.abs(ocv - v)
        ocv_index = np.argmin(delta)
        z = self.ocv_curve[ocv_index,1]
        return round(z, 4)  
    
    def ocv_inverse_exact_vectorized(self, v):
        """Returns the exact SOC using the expert-provided OCV curve in a vectorized manner.

        Args:
            v (numpy.ndarray):
                An array of OCV values for which SOC needs to be computed.

        Returns:
            numpy.ndarray:
                The required SOC for each input OCV value.
        """
        ocv = self.ocv_curve[:, 2]
        delta = np.abs(ocv - v[:, np.newaxis])  # Vectorized computation of delta
        ocv_indices = np.argmin(delta, axis=1)  # Find indices of minimum delta along axis 1
        z = self.ocv_curve[ocv_indices, 1]  # Extract corresponding SOC values
        return np.round(z, 4)    
    
    # --------------------------------------------- Custom Loss Functions --------------------------------------------
#     def soc_constraint(self, y_true, y_pred, scalers_train):
#         """Computes the custom SOC constraint.

#         Args:
#             y_true (numpy.ndarray): 
#                 The array of ground truth voltage values

#             y_pred (numpy.ndarray): 
#                 The array of predicted voltage values

#             scalers_train ((sklearn.preprocessing.MinMaxScaler, sklearn.preprocessing.MinMaxScaler)):
#                 A tuple of scaler objects used to scale and rescale X and y for training
#         Returns:
#             The SOC based loss. 
#         """
#         losses = []

#         for i in range(len(y_true)):
#             # SOC_hat
#             z_hat = self.ocv_inverse_exact(
#                         scalers_train.inverse_transform(
#                             np.array([y_true[i]]).reshape(1, -1))[0][0])
#             # SOC_label
#             z = self.ocv_inverse_exact(
#                         scalers_train.inverse_transform(
#                             np.array(
#                                 [y_pred.numpy()[i][0]]).reshape(1, -1))[0][0])


#             losses.append(backend.square(z_hat - z).numpy())

#         loss = tf.constant(losses, dtype=tf.float32)

#         # to visualize loss during training    
#         print(' - const:', round(np.mean(loss.numpy()), 4))
#         return loss

    def soc_constraint(self, y_true, y_pred, scalers_train):
        """Computes the custom SOC constraint.

        Args:
            y_true (numpy.ndarray): 
                The array of ground truth voltage values

            y_pred (tensorflow.Tensor): 
                The array of predicted voltage values (Tensorflow Tensor)

            scalers_train ((sklearn.preprocessing.MinMaxScaler, sklearn.preprocessing.MinMaxScaler)):
                A tuple of scaler objects used to scale and rescale X and y for training
        Returns:
            The SOC based loss. 
        """
        y_true_unscaled = scalers_train.inverse_transform(y_true.reshape(-1, 1))[:, 0]
        y_pred_unscaled = scalers_train.inverse_transform(y_pred.numpy().reshape(-1, 1))[:, 0]

        z_hat = self.ocv_inverse_exact_vectorized(y_true_unscaled)
        z = self.ocv_inverse_exact_vectorized(y_pred_unscaled)
        

        loss = tf.square(z_hat - z)

        # To visualize loss during training    
#         print(' - const:', tf.reduce_mean(loss).numpy())
        return loss

    def combine_losses(self, params, scalers_train):
        """Combines multiple custom loss functions.

        The loss functions provided by the 'loss_funcs' key in the params array will be combined in a weighted sum. 
        Weights are determined by the respective 'lambda_xyz' key. 

        Args:
            params (dict): 
                A dictionary containing the hyperparameters

            scalers_train ((sklearn.preprocessing.MinMaxScaler, sklearn.preprocessing.MinMaxScaler)):
                A tuple of scaler objects used to scale and rescale X and y for training
        Returns: 
            The combined loss value.
        """
        # wrapper function needed for the custom loss function to be accepted from keras
        def loss(data, y_pred):
            y_true = data[:,0]
            y_true = np.reshape(y_true, (-1, 1))

            accumulated_loss = 0
            loss_functions = params['loss_funcs']

#             # add custom loss function here 
#             if 'mse' in loss_functions:
#                 accumulated_loss += params['lambda_mse'] * losses.mean_squared_error(y_true, y_pred)
#             if 'soc' in loss_functions:
#                 accumulated_loss += params['lambda_soc'] * self.soc_constraint(y_true, y_pred, scalers_train)

            if 'mse' in loss_functions:
                mse_loss = params['lambda_mse'] * losses.mean_squared_error(y_true, y_pred)
                accumulated_loss += mse_loss
#                 print("mse loss dtype:", mse_loss.dtype)  # Add this line for debugging


            if 'soc' in loss_functions:
                soc_loss = params['lambda_soc'] * self.soc_constraint(y_true, y_pred, scalers_train)
                soc_loss = tf.cast(soc_loss, dtype=tf.float32)  # Cast soc_loss to float32
                accumulated_loss += soc_loss
#                 print("soc loss dtype:", soc_loss.dtype)  # Add this line for debugging

            #print(accumulated_loss.shape)
            return accumulated_loss
        return loss    
    
    
    def initialize(self, params, scalers_train):
        """Initializes the LSTM model.

        For visualization purposes, a summary of the model will be printed.

        Args:
            params (dict): 
                A dictionary containing the hyperparameters

            scalers_train ((sklearn.preprocessing.MinMaxScaler, sklearn.preprocessing.MinMaxScaler)):
                A tuple of scaler objects used to scale and rescale X and y for training
        """
        # --------- create model ---------
        model = tf.keras.Sequential(name='Loss_LSTM')

        # layer 1
        if params['n_lstm_layers'] == 1:
            model.add(layers.LSTM(units=params['n_lstm_units_1'], input_shape=(params['n_steps'], params['n_features']),
                                  return_sequences=False))
        # layer 2+
        else:
            model.add(layers.LSTM(units=params['n_lstm_units_1'], input_shape=(params['n_steps'], params['n_features']),
                                  return_sequences=True))
            for i in range(params['n_lstm_layers'] - 2):
                model.add(layers.LSTM(units=params['n_lstm_units_2'], return_sequences=True))
            model.add(layers.LSTM(units=params['n_lstm_units_2'], return_sequences=False))
            
            
        # output layer
        model.add(layers.Dense(1, activation=params['activation_output_layer']))
    
        # --------- compile model ---------
        custom_loss = self.combine_losses(params, scalers_train) # define custom loss
        optimizer = optimizers.Adam(learning_rate=params['lr'])

        model.compile(run_eagerly=True, optimizer=optimizer, loss=custom_loss, metrics=['mse', params['metric']])
        
        # save model parameters
        self.model = model
        self.params = params
        
        
        self.ocv_curve = np.loadtxt('../../../data/processed/soc_ocv/ocv_curve.csv', delimiter=';')
        
        return None


    def plot_training_results(self, history):
        # Extract training and validation metrics
        loss = history['loss']
        mae = history['mae']

        epochs = range(1, len(loss) + 1)

        # Create subplots for loss and metrics
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5), dpi=140)

        # Plot training and validation loss
        ax1.plot(epochs, loss, '-o', color='green', label='Training Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        fig.tight_layout()
        if 'results_path' in self.params: 
            fig.savefig(f"{self.params['results_path']}/learning_curve.png", dpi=140)
        else:
            plt.show()
        plt.close("all")
        plt.clf()    
        
        
    def train_f(self, X, y, scalers_train, verbose=0):
        """Trains the LSTM model.

        For visualization purposes, the MSE and MAE over all training epochs will be ploted.

        Args:
            X list of multiple (numpy.ndarray): 
                The input data
                
            y list of multiple (numpy.ndarray): 
                The groundtruth output data
            
            scalers_train ((sklearn.preprocessing.MinMaxScaler, sklearn.preprocessing.MinMaxScaler)):
                A tuple of scaler objects used to scale and rescale X and y for training
                
        Returns:
            The time_callback which is used to measure the time needed to train the model. 
            In adition the matplotlib figure used to plot the visualization. 
            This is needed so that the plots can be saved at the appropriate location.
        """
        # --------- train model ---------
        

  
        n_samples = len(X)
        n_epochs = self.params['n_epochs']
               
        history = {'loss': [], 'mae': []}

        for epoch in range(n_epochs):
            # print(f"Epoch {epoch + 1}/{n_epochs}")
            epoch_losses = []
            epoch_mae = []
            sample_indices = np.random.permutation(n_samples)

            for i in sample_indices:
                hist = self.model.fit(X[i], y[i], epochs=1, batch_size=32, shuffle=False, verbose=0)

                epoch_losses.extend(hist.history['loss'])
                epoch_mae.extend(hist.history['mae'])

            avg_loss = np.mean(epoch_losses)
            avg_mae = np.mean(epoch_mae)
            history['loss'].append(avg_loss)
            history['mae'].append(avg_mae)
            #print(f'epoch: {epoch}, loss: {avg_loss}')


        self.plot_training_results(history)
        self.model.save(self.params['model_save_path'] + "/model.h5")

        # save parameters
        self.history = history
        self.scalers_train = scalers_train        

   

    def test_usecases(self, X_case_1, y_case_1, X_case_2, y_case_2, X_case_3, y_case_3, scalers_train):
        """Tests the LSTM model on validation and test data.
    
        For visualization purposes, a table with several metrics for training, validation and test data 
        will be printed in adition to plots of the validation and test profiles used.
 
        Args:
            X_case_1 (numpy.ndarray):
                Use case 1 input data
                
            y_case_1 (numpy.ndarray): 
                Use case 1 output data
                
            X_case_2 (numpy.ndarray): 
                Use case 2 input data
                
            y_case_2 (numpy.ndarray): 
                Use case 2 output data
                
            X_case_3 (numpy.ndarray): 
                Use case 3 input data
                
            y_case_3 (numpy.ndarray): 
                Use case 3 output data
                
            scalers_train ((sklearn.preprocessing.MinMaxScaler, sklearn.preprocessing.MinMaxScaler)):
                A tuple of scaler objects used to scale and rescale X and y for training
        """
        # --------- predict on data ---------
        yhat_case_1 = self.model.predict(X_case_1, verbose=0)
        yhat_case_1_unscaled = scalers_train.inverse_transform(yhat_case_1)
        y_case_1_unscaled = scalers_train.inverse_transform(y_case_1)

        yhat_case_2 = self.model.predict(X_case_2, verbose=0)
        yhat_case_2_unscaled = scalers_train.inverse_transform(yhat_case_2)
        y_case_2_unscaled = scalers_train.inverse_transform(y_case_2)

        yhat_case_3 = self.model.predict(X_case_3, verbose=0)
        yhat_case_3_unscaled = scalers_train.inverse_transform(yhat_case_3)
        y_case_3_unscaled = scalers_train.inverse_transform(y_case_3)

        # --------- compute error ---------
        case_1_rmse = metrics.mean_squared_error(y_case_1_unscaled, yhat_case_1_unscaled, squared=False) * 1000
        case_2_rmse = metrics.mean_squared_error(y_case_2_unscaled, yhat_case_2_unscaled, squared=False) * 1000
        case_3_rmse = metrics.mean_squared_error(y_case_3_unscaled, yhat_case_3_unscaled, squared=False) * 1000

        case_1_mae = metrics.mean_absolute_error(y_case_1_unscaled, yhat_case_1_unscaled) * 1000
        case_2_mae = metrics.mean_absolute_error(y_case_2_unscaled, yhat_case_2_unscaled) * 1000
        case_3_mae = metrics.mean_absolute_error(y_case_3_unscaled, yhat_case_3_unscaled) * 1000

        results = {
            'use_case_1_rmse': case_1_rmse,
            'use_case_1_mae': case_1_mae,
            'use_case_2_rmse': case_2_rmse,
            'use_case_2_mae': case_2_mae,
            'use_case_3_rmse': case_3_rmse,
            'use_case_3_mae': case_3_mae,
            'avg_rmse': (case_1_rmse + case_2_rmse + case_3_rmse) / 3,
            'avg_mae': (case_1_mae + case_2_mae + case_3_mae) / 3
        }

        # Save results to a JSON file
        with open(self.params['results_path'] + '/use_case_results.json', 'w') as file:
            json.dump(results, file, indent=4)

        # --------- plot results ---------
        fig, _ = plt.subplots(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        time_case_1 = np.arange(yhat_case_1_unscaled.shape[0] * self.params['d_sample'])[
                      ::self.params['d_sample']] * 0.25
        plt.plot(time_case_1, yhat_case_1_unscaled, color='blue', label='predicted')
        plt.plot(time_case_1, y_case_1_unscaled, color='g', dashes=[2, 2], label='measured')
        plt.fill_between(time_case_1, yhat_case_1_unscaled.flatten(), y_case_1_unscaled.flatten(), label='delta',
                         color='lightgrey')
        plt.ylabel('voltage (V)', fontsize=20)
        plt.xlabel('time (s)', fontsize=20)
        plt.title('Reproduction', fontsize=20)
        plt.legend()
        plt.subplot(1, 3, 2)
        time_case_2 = np.arange(yhat_case_2_unscaled.shape[0] * self.params['d_sample'])[
                      ::self.params['d_sample']] * 0.25
        plt.plot(time_case_2, yhat_case_2_unscaled, color='blue', label='predicted')
        plt.plot(time_case_2, y_case_2_unscaled, color='g', dashes=[2, 2], label='measured')
        plt.fill_between(time_case_2, yhat_case_2_unscaled.flatten(), y_case_2_unscaled.flatten(), label='delta',
                         color='lightgrey')
        plt.xlabel('time (s)', fontsize=20)
        plt.title('Abstraction', fontsize=20)
        plt.legend()
        plt.subplot(1, 3, 3)
        time_case_3 = np.arange(yhat_case_3_unscaled.shape[0] * self.params['d_sample'])[
                      ::self.params['d_sample']] * 0.25
        plt.plot(time_case_3, yhat_case_3_unscaled, color='blue', label='predicted')
        plt.plot(time_case_3, y_case_3_unscaled, color='g', dashes=[2, 2], label='measured')
        plt.fill_between(time_case_3, yhat_case_3_unscaled.flatten(), y_case_3_unscaled.flatten(), label='delta',
                         color='lightgrey')
        plt.xlabel('time (s)', fontsize=20)
        plt.title('Generalization', fontsize=20)
        plt.legend()
        fig.tight_layout()
        if 'results_path' in self.params: 
            fig.savefig(f"{self.params['results_path']}/use_cases.png")
        else:
            plt.show()
        plt.close("all")
        plt.clf()    
