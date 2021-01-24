"""
Module containing different utility functions used for preprocessing time series data.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage 
from deprecated import deprecated

def subsequences(sequence_X, sequence_y, n_steps):
    """Creates subsequences of the original sequences to fit the keras model structure.
 
    Args:
        sequence_1 (numpy.ndarray): 
            The first sequence which gets converted into multiple subarrays of length: n_steps
            
        sequence_2 (numpy.ndarray): 
            The second sequence, each n_steps'th element will be part of the output array
            
        n_steps (int): 
            The amount of time steps used as an input into the LSTM for prediction

    Returns:
        A tuple of 2 numpy arrays in the required format.

        X.shape = (sequence_X.shape[0] - n_steps, n_steps)
        y.shape = (sequence_y.shape[0] - n_steps, 1)

    Raises:
        Exception: If n_steps exceeds the length of sequence_X no subsequences can be created
    """
    
    if n_steps > len(sequence_X):
        raise Exception('data_preprocessing.subsequences: n_steps should not exceed the sequence length')
    
    sequence_X = np.append(np.repeat(sequence_X[0], n_steps - 1), sequence_X)
    
    X, y = list(), list()
    for i in range(len(sequence_X)):
        end_ix = i + n_steps

        if end_ix > len(sequence_X):
            break

        X.append(sequence_X[i:end_ix])
        y.append(sequence_y[i])
        
    return np.array(X), np.array(y)


def subsample(sequence, d_sample):
    """Subsamples the sequence data to decrease the amount of data points.

    Args:
        sequence (numpy.ndarray): 
            The input array to be subsampled
            
        d_sample (int): 
            The sample frequency, meaning every d_sample'th element will be part of the output

    Returns:
        The subsampled array.
    """
    return sequence[::d_sample]


def smooth(sequence, sigma):
    """Smooths the sequence data to decrease distortion created by measurement noise.

    Args: 
        sequence (numpy.ndarray): 
            The input array to be smoothed
            
        sigma (int): 
            The parameter for gaussian filtering

    Returns:
        The smoothed array
    """
    return ndimage.filters.gaussian_filter(sequence, sigma)


def align(sequence_1, sequence_2):
    """Aligns two sequences.

    In this context this means subsampling the first array so that it afterwards has the same size as the second array.
    
    Args: 
        sequence_1 (numpy.ndarray): 
            The arrray to be aligned (the one with bigger size)
            
        sequence_2 (numpy.ndarray): 
            The array to be aligned to

    Returns:
        The algined array
        
    Raises:
        Exception: If the array which is being aligned is of smaller size than the one which it is supposed to be aligned to.
    """
    
    if len(sequence_1) < len(sequence_2):
        raise Exception('data_preprocessing.align: missmatch of sequence lengths')
    
    sample_ratio = sequence_1.shape[0] / sequence_2.shape[0]

    aligned_sequence = list()
    for i in range(len(sequence_2)):
        aligned_sequence.append(sequence_1[int(np.round(i * sample_ratio))])

    aligned_sequence = np.array(aligned_sequence)
    
    return aligned_sequence


def preprocess_raw_data(params, sequence):
    """Preprocesses the raw sequence by subsamling, smoothing and scaling the data.

    Args:
        params (dict): 
            Dictionary containing the keys 's_sample' and 'gauss_sigma' which represent the input parameter for preprocessing
            
        sequence (numpy.ndarray): 
            The sequence to be preprocessed
        
    Returns:
        A tuple of 2 values. The preprocessed sequence and the scaler object (used for retransforming after training)
    """
    
    sequence = subsample(sequence, params['d_sample'])
    sequence = smooth(sequence, params['gauss_sigma'])
    sequence = np.reshape(sequence, (-1, 1))
    
    scaler = MinMaxScaler(feature_range = (-1, 1))
    scaler.fit(sequence)
    sequence = scaler.transform(sequence)
    
    return sequence, scaler


def load_current_raw_data(profile):
    """Loads the current raw data.

    This method will extract raw data provided by the inverter current sensor.

    Args:
        profile (str):
            The FOBSS profile from which the data should be loaded
        
    Returns:
        A numpy array containing the requested raw data.
    """
    
    current_data = np.loadtxt('../data/fobss_data/data/' + profile + '/inverter/Inverter_Current.csv', delimiter=';')
    current_data = current_data[:,1] # only the first column includes necessary information
    return current_data


def load_voltage_raw_data(profile, slave, cell):
    """ Loads the voltage raw data.

    This method will extract raw data provided by the specified cell voltage sensor.

    Args:
        profile (str): 
            The FOBSS profile from which the data should be loaded
        slave (int): 
            The battery slave (or stack)
        cell (int): 
            The battery cell 
        
    Returns:
        A numpy array containing the requested raw data.
    """
    voltage_data = np.loadtxt('../data/fobss_data/data/' + profile + '/cells/Slave_' + str(slave) + '_Cell_Voltages.csv', delimiter=';')
    voltage_data = voltage_data[:,cell] # select correct cell out of slave data
    return voltage_data


def prepare_data(params, profiles, slave, cell):
    """Prepares the requested data to be suitable for the network.

    Args:
        params (dict): 
            Dictionary containing the keys 's_sample', 'gauss_sigma', 'n_steps'
            
        profiles (list): 
            A list of all FOBSS profiles which should be used
            
        slave (int): 
            The battery slave (or stack)
        
        cell (int): 
            The battery cell 

    Returns:
        A tuple containing 3 values. The prepared input X, the prepared output/label y and the used scalers.
    """
    
    current_raw, voltage_raw = [], []
    for profile in profiles:
        current_raw = np.append(current_raw, load_current_raw_data(profile), axis=0)
        voltage_raw = np.append(voltage_raw, load_voltage_raw_data(profile, slave, cell), axis=0)
    
    current_cum = np.cumsum(current_raw)

    # preprocess data
    current_preprocessed, scaler_cur = preprocess_raw_data(params, current_raw)
    current_cum_preprocessed, scaler_cur_cum = preprocess_raw_data(params, current_cum)
    voltage_preprocessed, scaler_volt = preprocess_raw_data(params, voltage_raw)

    # align current sequence to voltage if sample frequency differs
    if voltage_preprocessed.shape[0] != current_preprocessed.shape[0]:
        current_preprocessed = align(current_preprocessed, voltage_preprocessed)
        current_cum_preprocessed = align(current_cum_preprocessed, voltage_preprocessed)

    # create input features
    X1, y = subsequences(current_preprocessed, voltage_preprocessed, params['n_steps'])
    y = np.reshape(y, (-1, 1))
    X1 = X1.reshape(X1.shape[0], X1.shape[1], 1)
    
    X2, _ = subsequences(current_cum_preprocessed, voltage_preprocessed, params['n_steps'])
    X2 = X2.reshape(X2.shape[0], X2.shape[1], 1)

    X = np.append(X1, X2, axis=2)
    print('Input:', X.shape, '\nOutput/Label:', y.shape)
    
    scalers = scaler_cur, scaler_volt
    
    return X, y, scalers

def prepare_hybrid_data(params, profiles, slave, cell):
    current_raw, voltage_raw = [], []
    for profile in profiles:
        current_raw = np.append(current_raw, load_current_raw_data(profile), axis=0)
        voltage_raw = np.append(voltage_raw, load_voltage_raw_data(profile, slave, cell), axis=0)
    
    current_cum = np.cumsum(current_raw)
    current_preprocessed, scaler_cur = preprocess_raw_data(params, current_raw)
    current_cum_preprocessed, scaler_cur_cum = preprocess_raw_data(params, current_cum)
    voltage_preprocessed, scaler_volt = preprocess_raw_data(params, voltage_raw)

    if voltage_preprocessed.shape[0] != current_preprocessed.shape[0]:
        current_preprocessed = align(current_preprocessed, voltage_preprocessed)
        current_cum_preprocessed = align(current_cum_preprocessed, voltage_preprocessed)

    X1, y = subsequences(current_preprocessed, voltage_preprocessed, params['n_steps'])
    y = np.reshape(y, (-1, 1))
    X1 = X1.reshape(X1.shape[0], X1.shape[1], 1)
    X2, _ = subsequences(current_cum_preprocessed, voltage_preprocessed, params['n_steps'])
    X2 = X2.reshape(X2.shape[0], X2.shape[1], 1)
    
    # add voltage computed by theory-based model
    theory_data = np.load('trained_models/TGDS/9079/predictions.npy') # TODO: change this to the appropriate EC-Model path
    theory_preprocessed, scaler_res = preprocess_raw_data(params, theory_data)
    
    X3, _ = subsequences(theory_preprocessed, voltage_preprocessed, params['n_steps'])
    X3 = X3.reshape(X3.shape[0], X3.shape[1], 1)
    
    X = np.append(X1, X2, axis=2)
    X = np.append(X, X3, axis=2)
    print('Input:', X.shape, '\nOutput/Label:', y.shape)
    
    scalers = scaler_cur, scaler_volt
    
    return X, y, scalers


@deprecated(reason="data_preprocessing.preprocess_raw_data should be used instead")
def prepare(input_sequence, label_sequence, aligned, d_sample, n_steps, sigma):
    """Prepares the data for input into the LSTM.

    Preparation incudes subsampling, smoothing, aligning differnt sized sequences and reshaping the sequence to the requested format.
    
    Args:
        input_sequence (numpy.ndarray): 
            The input feature sequence
            
        label_sequence (numpy.ndarray): 
            The output/groud truth sequence
            
        aligned (bool): 
            Indicates if input and label sequence are of equal size or need alignment
            
        d_sample (int): 
            Sample frequency
            
        n_steps (int): 
            The amount of time steps used as an input into the LSTM for prediction
            
        sigma (int): 
            Parameter for the data smoothing

    Returns:
        A tuple of 3 values. The prepared input sequence X, the output sequence of labels y and the scaler component for y. 
        This is needed afterwards to scale the output back to the original value range.
    """
    
    # align data if not of equal size
    if not aligned:        
        input_sequence = align(input_sequence, label_sequence)

    # subsample and smooth data 
    input_sequence_ = subsample(input_sequence, d_sample)
    input_sequence_ = smooth(input_sequence_, sigma)
    
    label_sequence_ = subsample(label_sequence, d_sample)
    label_sequence_ = smooth(label_sequence_, sigma)

    # convert into X and y sequences
    X, y = subsequences(input_sequence_, label_sequence_, n_steps)
    y = np.reshape(y, (-1, 1))

    # fit and scale X
    scaler_X = MinMaxScaler(feature_range = (0, 1))
    scaler_X.fit(X)
    X_scaled = scaler_X.transform(X)
    
    # fit and scale y
    scaler_y = MinMaxScaler(feature_range = (0, 1))
    scaler_y.fit(y)
    y_scaled = scaler_y.transform(y)
    
    # reshape into correct format
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    
    return X_scaled, y_scaled, scaler_y
