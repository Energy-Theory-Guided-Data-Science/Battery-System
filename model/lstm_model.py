import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from tensorflow import keras
from tensorflow.keras import layers
from tabulate import tabulate

class Model: 
    """ Initializes the LSTM model 

        Visualization: a summary of the model will be printed
        
    Args:
        params: dictionary containing the model hyperparameters

    Returns:
        The initialized model.

    """
    def initialize(self, params):
        # --------- create model ---------
        model = keras.Sequential()
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
        model.compile(optimizer = params['optimizer'], loss = params['loss'], metrics=[params['metric']])
        
        # save parameters
        self.model = model
        self.params = params
        return model

    
    """ Trains the LSTM model 
    
        Visualization: illustrates the training loss and absolute error over all training epochs
 
    Args:
        X: the input data
        y: the groundtruth output data

    Returns:
        The training history provided by keras

    """
    def train(self, X, y, scalers):
        history = self.model.fit(X, y, epochs = self.params['n_epochs'], verbose = 1)
        
        # plot loss and error
        loss = history.history['loss']
        metrics = history.history['mae']
        epochs = range(1,len(loss)+1)
        
        plt.subplots(figsize = (5,5))
        plt.subplot(2,1,1)
        plt.plot(epochs,loss,'-o',label='training loss')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(epochs,metrics,'-o', color='green',label='absolute error')
        plt.legend()
        
        # save parameters
        self.history = history
        self.scalers_train = scalers
        return history

    """ Tests the LSTM model on validation and test data
    
        Validation: shows the MSE of training, validation and test data and prints the test profiles
 
    Args:
        X_validation: validation input data
        y_validation: validation output data
        X_test: test input data
        y_test: test output data

    Returns:
        Tuple containin training, validation and test error (MSE)
    """
    def test(self, X_train, y_train, X_validation, y_validation, X_test, y_test, scalers):
        yhat_train = self.model.predict(X_train, verbose = 1)
        yhat_train_unscaled = scalers[0][1].inverse_transform(yhat_train)
        y_train_unscaled = scalers[0][1].inverse_transform(y_train)
        
        yhat_validation = self.model.predict(X_validation, verbose = 1)
        yhat_validation_unscaled = scalers[1][1].inverse_transform(yhat_validation)
        y_validation_unscaled = scalers[1][1].inverse_transform(y_validation)
        
        yhat_test = self.model.predict(X_test, verbose = 1)
        yhat_test_unscaled = scalers[2][1].inverse_transform(yhat_test)
        y_test_unscaled = scalers[2][1].inverse_transform(y_test)

        # compute train, test and validation error
        train_mse = metrics.mean_squared_error(y_train_unscaled, yhat_train_unscaled)
        validation_mse = metrics.mean_squared_error(y_validation_unscaled, yhat_validation_unscaled)
        test_mse = metrics.mean_squared_error(y_test_unscaled, yhat_test_unscaled)
        
        train_mae = metrics.mean_absolute_error(y_train_unscaled, yhat_train_unscaled)
        validation_mae = metrics.mean_absolute_error(y_validation_unscaled, yhat_validation_unscaled)
        test_mae = metrics.mean_absolute_error(y_test_unscaled, yhat_test_unscaled)
        
        train_max = metrics.max_error(y_train_unscaled, yhat_train_unscaled)
        validation_max = metrics.max_error(y_validation_unscaled, yhat_validation_unscaled)
        test_max = metrics.max_error(y_test_unscaled, yhat_test_unscaled)
        
        print('###########################################################')
        error_table = tabulate([['MSE', round(train_mse, 6), round(validation_mse, 6), round(test_mse, 6)], 
          ['MAE', round(train_mae, 4), round(validation_mae, 4), round(test_mae, 4)], 
          ['MaxE', round(train_max, 4), round(validation_max, 4), round(test_max, 4)]], headers=['Training', 'Validation', 'Test'])
        print(error_table)
        print('###########################################################')

        # plot profiles results
        plt.subplots(figsize = (7,10))
        plt.subplot(2,1,1)  
        plt.plot(yhat_validation_unscaled, color='red', label = 'predicted')
        plt.plot(y_validation_unscaled, color='blue', label = 'measured')
        plt.title('Validation Data')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(yhat_test_unscaled, color='red', label = 'predicted')
        plt.plot(y_test_unscaled, color='blue', label = 'measured')
        plt.title('Test Data')
        plt.legend()
        plt.show()
        
        return train_mse, validation_mse, test_mse
        