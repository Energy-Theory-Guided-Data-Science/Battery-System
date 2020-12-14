import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error

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
        X_validation
        y_validation
        X_test
        y_test

    Returns:
        Tuple containin training, validation and test error (MSE)
    """
    def test(self, X_validation, y_validation, X_test, y_test, scalers):
#         yhat1 = self.model.predict(X_test1, verbose = 1)
#         yhat_rescaled1 = scaler[0].inverse_transform(yhat1)
#         y_test_unscaled1 = scaler[0].inverse_transform(y_test1)

#         yhat2 = self.model.predict(X_test2, verbose = 1)
#         yhat_rescaled2 = scaler[1].inverse_transform(yhat2)
#         y_test_unscaled2 = scaler[1].inverse_transform(y_test2)

#         yhat3 = self.model.predict(X_test3, verbose = 1)
#         yhat_rescaled3 = scaler[2].inverse_transform(yhat3)
#         y_test_unscaled3 = scaler[2].inverse_transform(y_test3)

        yhat_validation = self.model.predict(X_validation, verbose = 1)
        yhat_validation_unscaled = scalers[0].inverse_transform(yhat_validation)
        y_validation_unscaled = scalers[0].inverse_transform(y_validation)
        
        yhat_test = self.model.predict(X_test, verbose = 1)
        yhat_test_unscaled = scalers[1].inverse_transform(yhat_test)
        y_test_unscaled = scalers[1].inverse_transform(y_test)

        # compute train, test and validation error
        train_error = self.history.history['loss'][-1]
        validation_error = mean_squared_error(y_validation_unscaled, yhat_validation_unscaled)
        test_error = mean_squared_error(y_test_unscaled, yhat_test_unscaled)

        print('Train error:', round(train_error, 6))
        print('Validation error:', round(validation_error, 6))
        print('Test error', round(test_error, 6))

        # plot profiles results
        plt.subplots(figsize = (7,10))
        plt.subplot(2,1,1)
        plt.plot(yhat_validation_unscaled, color='red', label = 'predicted')
        plt.plot(y_validation_unscaled, color='blue', label = 'measured')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(yhat_test_unscaled, color='red', label = 'predicted')
        plt.plot(y_test_unscaled, color='blue', label = 'measured')
        plt.legend()
#         plt.subplot(3,1,3)
#         plt.plot(yhat_rescaled2, color='red', label = 'predicted')
#         plt.plot(y_test_unscaled2, color='blue', label = 'measured')
#         plt.legend()
        plt.show()
        
        return train_error, validation_error, test_error
        