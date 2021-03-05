"""
Module containing different utility functions used for preprocessing time series data.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage 
from deprecated import deprecated
import matplotlib.pyplot as plt

import src.models.thevenin_model as thevenin

# ---------------------------------------------- Load Data -------------------------------------------------------
def load_current_raw_data(profile):
    """Loads the current raw data.

    This method will extract raw data provided by the inverter current sensor.

    Args:
        profile (str):
            The FOBSS profile from which the data should be loaded
        
    Returns:
        A numpy array containing the requested raw data.
    """
    current_data = np.loadtxt('../../../data/raw/fobss_data/data/' + profile + '/inverter/Inverter_Current.csv', delimiter=';')
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
    voltage_data = np.loadtxt('../../../data/raw/fobss_data/data/' + profile + '/cells/Slave_' + str(slave) + '_Cell_Voltages.csv', delimiter=';')
    voltage_data = voltage_data[:,cell] # select correct cell out of slave data
    return voltage_data

# ----------------------------------------- Data Preprocessing --------------------------------------------------
def subsequences(sequence_X, sequence_y, n_steps):
    """Creates subsequences of the original sequences to fit the keras model structure.
    
    It is recommended to align the sequences first with util.align().
 
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
    
    scaler = MinMaxScaler(feature_range = (params['feature_range_low'], params['feature_range_high']))
    scaler.fit(sequence)
    sequence = scaler.transform(sequence)
    return sequence, scaler


def preprocess_raw_current(params, sequence):
    sequence = subsample(sequence, params['d_sample'])
    sequence = smooth(sequence, params['gauss_sigma'])
    sequence = np.reshape(sequence, (-1, 1))
    
    scaler = MinMaxScaler(feature_range=(params['feature_range_cur_low'], params['feature_range_cur_high']))
    
    boundaries = [params['boundary_cur_low'], params['boundary_cur_high']]
    boundaries = np.reshape(boundaries, (-1, 1))
    scaler.fit(boundaries)
    
    sequence = scaler.transform(sequence)
    return sequence, scaler


def preprocess_raw_acc_cur(params, sequence):
    sequence = subsample(sequence, params['d_sample'])
    sequence = smooth(sequence, params['gauss_sigma'])
    sequence = np.reshape(sequence, (-1, 1))
    
    scaler = MinMaxScaler(feature_range=(params['feature_range_acc_cur_low'], params['feature_range_acc_cur_high']))
    
    boundaries = [params['boundary_acc_cur_low'], params['boundary_acc_cur_high']]
    boundaries = np.reshape(boundaries, (-1, 1))
    scaler.fit(boundaries)
    
    sequence = scaler.transform(sequence)
    return sequence, scaler


def preprocess_raw_charge(params, sequence):
    sequence = np.reshape(sequence, (-1, 1))
    sequence = subsample(sequence, params['d_sample'])

    scaler = MinMaxScaler(feature_range=(params['feature_range_charge_low'], params['feature_range_charge_high']))
    
    boundaries = [params['boundary_charge_low'], params['boundary_charge_high']]
    boundaries = np.reshape(boundaries, (-1, 1))
    scaler.fit(boundaries)
    
    sequence = scaler.transform(sequence)
    return sequence, scaler


def preprocess_raw_voltage(params, sequence):
    sequence = subsample(sequence, params['d_sample'])
    sequence = smooth(sequence, params['gauss_sigma'])
    sequence = np.reshape(sequence, (-1, 1))
    
    scaler = MinMaxScaler(feature_range=(params['feature_range_volt_low'], params['feature_range_volt_high']))
    
    boundaries = [params['boundary_voltage_low'], params['boundary_voltage_high']]
    boundaries = np.reshape(boundaries, (-1, 1))
    scaler.fit(boundaries)
    
    sequence = scaler.transform(sequence)
    
    return sequence, scaler

def preprocess_delta_voltage(params, sequence):
    sequence = subsample(sequence, params['d_sample'])
    sequence = smooth(sequence, params['gauss_sigma'])
    sequence = np.reshape(sequence, (-1, 1))
    
    scaler = MinMaxScaler(feature_range=(params['feature_range_delta_low'], params['feature_range_delta_high']))
    
    boundaries = [params['boundary_delta_low'], params['boundary_delta_high']]
    boundaries = np.reshape(boundaries, (-1, 1))
    scaler.fit(boundaries)
    
    sequence = scaler.transform(sequence)
    
    return sequence, scaler

# ---------------------------------------- Prepare Network Input -------------------------------------------------
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
    i = 0
    
    for profile in profiles:
        # load data
        current_raw = load_current_raw_data(profile)
        current_cum = np.cumsum(current_raw)
        current_cum = current_cum / np.max(np.abs(current_cum))
        voltage_raw = load_voltage_raw_data(profile, slave, cell)

        # preprocess data
        current_preprocessed, _ = preprocess_raw_current(params, current_raw)    
        current_cum_preprocessed, _ = preprocess_raw_acc_cur(params, current_cum)
        voltage_preprocessed, scaler_volt = preprocess_raw_voltage(params, voltage_raw)

        # align current sequence to voltage if sample frequency differs
        if voltage_preprocessed.shape[0] != current_preprocessed.shape[0]:
            current_preprocessed = align(current_preprocessed, voltage_preprocessed)
            current_cum_preprocessed = align(current_cum_preprocessed, voltage_preprocessed)

        # create input features
        profile_X1, profile_y = subsequences(current_preprocessed, voltage_preprocessed, params['n_steps'])
        profile_y = np.reshape(profile_y, (-1, 1))
        profile_X1 = profile_X1.reshape(profile_X1.shape[0], profile_X1.shape[1], 1)

        profile_X2, _ = subsequences(current_cum_preprocessed, voltage_preprocessed, params['n_steps'])
        profile_X2 = profile_X2.reshape(profile_X2.shape[0], profile_X2.shape[1], 1)

        profile_X = np.append(profile_X1, profile_X2, axis=2)

        # append for multiple profiles
        if (i == 0):
            X = profile_X
            y = profile_y
        else:
            X = np.append(X, profile_X, axis=0)
            y = np.append(y, profile_y, axis=0)
        i += 1
    
    print('Input:', X.shape, ', Output/Label:', y.shape)
    return X, y, scaler_volt


def prepare_current_input(params, profiles, slave, cell):
    current_raw, voltage_raw = [], []
    for profile in profiles:
        current_raw = np.append(current_raw, load_current_raw_data(profile), axis=0)
        voltage_raw = np.append(voltage_raw, load_voltage_raw_data(profile, slave, cell), axis=0)
    
    # preprocess data
    current_preprocessed, _ = preprocess_raw_current(params, current_raw)    
    voltage_preprocessed, scaler_volt = preprocess_raw_current(params, voltage_raw)

    # align current sequence to voltage if sample frequency differs
    if voltage_preprocessed.shape[0] != current_preprocessed.shape[0]:
        current_preprocessed = align(current_preprocessed, voltage_preprocessed)

    # create input features
    X, y = subsequences(current_preprocessed, voltage_preprocessed, params['n_steps'])
    y = np.reshape(y, (-1, 1))
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    print('Input:', X.shape, '\nOutput/Label:', y.shape)
        
    return X, y, scaler_volt


def prepare_current_charge_input(params, profiles, slave, cell):
    i = 0
    
    for profile in profiles:
        # load data
        current_raw = load_current_raw_data(profile)
        
        charge_raw = []
        q_t = 0
        for j in range(len(current_raw)):
            q_t += current_raw[j] * params['d_t']  / 3600
            charge_raw.append(q_t)
        
        voltage_raw = load_voltage_raw_data(profile, slave, cell)
        
        # preprocess data
        current_preprocessed, _ = preprocess_raw_current(params, current_raw)    
        charge_preprocessed, _ = preprocess_raw_charge(params, charge_raw)
        voltage_preprocessed, scaler_volt = preprocess_raw_voltage(params, voltage_raw)

        # align current sequence to voltage if sample frequency differs
        if voltage_preprocessed.shape[0] != current_preprocessed.shape[0]:
            current_preprocessed = align(current_preprocessed, voltage_preprocessed)
            charge_preprocessed = align(charge_preprocessed, voltage_preprocessed)

        # create input features
        profile_X1, profile_y = subsequences(current_preprocessed, voltage_preprocessed, params['n_steps'])
        profile_y = np.reshape(profile_y, (-1, 1))
        profile_X1 = profile_X1.reshape(profile_X1.shape[0], profile_X1.shape[1], 1)

        profile_X2, _ = subsequences(charge_preprocessed, voltage_preprocessed, params['n_steps'])
        profile_X2 = profile_X2.reshape(profile_X2.shape[0], profile_X2.shape[1], 1)

        profile_X = np.append(profile_X1, profile_X2, axis=2)

        # append for multiple profiles
        if (i == 0):
            X = profile_X
            y = profile_y
        else:
            X = np.append(X, profile_X, axis=0)
            y = np.append(y, profile_y, axis=0)
        i += 1
    
    print('Input:', X.shape, ', Output/Label:', y.shape)
    return X, y, scaler_volt


def prepare_current_charge_delta_input(params, profiles, slave, cell):
    i = 0
    
    for profile in profiles:
        # load data
        current_raw = load_current_raw_data(profile)
        
        charge_raw = []
        q_t = 0
        for j in range(len(current_raw)):
            q_t += current_raw[j] * params['d_t']  / 3600
            charge_raw.append(q_t)
        
        voltage_raw = load_voltage_raw_data(profile, slave, cell)
        voltage_delta = voltage_raw - voltage_raw[0] # only predict voltage delta
        initial_voltage = np.repeat(voltage_raw[0], len(voltage_raw))
                
        # preprocess data
        current_preprocessed, _ = preprocess_raw_current(params, current_raw)    
        charge_preprocessed, _ = preprocess_raw_charge(params, charge_raw)
        voltage_preprocessed, scaler_volt = preprocess_delta_voltage(params, voltage_delta)
        intial_voltage_preprocessed, _ = preprocess_raw_voltage(params, initial_voltage)

        # align current sequence to voltage if sample frequency differs
        if voltage_preprocessed.shape[0] != current_preprocessed.shape[0]:
            current_preprocessed = align(current_preprocessed, voltage_preprocessed)
            charge_preprocessed = align(charge_preprocessed, voltage_preprocessed)

        # create input features
        profile_X1, profile_y = subsequences(current_preprocessed, voltage_preprocessed, params['n_steps'])
        profile_y = np.reshape(profile_y, (-1, 1))
        profile_X1 = profile_X1.reshape(profile_X1.shape[0], profile_X1.shape[1], 1)

        profile_X2, _ = subsequences(charge_preprocessed, voltage_preprocessed, params['n_steps'])
        profile_X2 = profile_X2.reshape(profile_X2.shape[0], profile_X2.shape[1], 1)
        
        profile_X3, _ = subsequences(intial_voltage_preprocessed, voltage_preprocessed, params['n_steps'])
        profile_X3 = profile_X3.reshape(profile_X3.shape[0], profile_X3.shape[1], 1)

        profile_X = np.append(profile_X1, profile_X2, axis=2)
        profile_X = np.append(profile_X, profile_X3, axis=2)

        # append for multiple profiles
        if (i == 0):
            X = profile_X
            y = profile_y
        else:
            X = np.append(X, profile_X, axis=0)
            y = np.append(y, profile_y, axis=0)
        i += 1
    
    print('Input:', X.shape, ', Output/Label:', y.shape)
    return X, y, scaler_volt

def prepare_hybrid_input(params, profiles, slave, cell):
    i = 0
    
    for profile in profiles:
        # load data
        current_raw = load_current_raw_data(profile)
        
        charge_raw = []
        q_t = 0
        for j in range(len(current_raw)):
            q_t += current_raw[j] * params['d_t']  / 3600
            charge_raw.append(q_t)
        
        voltage_raw = load_voltage_raw_data(profile, slave, cell)
        
        # predict on theory model
        thevenin_params = np.load('../../../models/T/theory_baseline-' + str(params['theory_model']) + '-parameters.npy', allow_pickle=True)
        thevenin_params = thevenin_params.item()

        thevenin_hyperparams = np.load('../../../models/T/theory_baseline-' + str(params['theory_model']) + '-hyperparameters.npy', allow_pickle=True)
        thevenin_hyperparams = thevenin_hyperparams.item()

        theory_raw = thevenin.predict(profile, thevenin_params['r_0'], thevenin_params['r_1'], thevenin_params['c_1'], thevenin_hyperparams)
        
        
        # preprocess data
        current_preprocessed, _ = preprocess_raw_current(params, current_raw)    
        charge_preprocessed, _ = preprocess_raw_charge(params, charge_raw)
        voltage_preprocessed, scaler_volt = preprocess_raw_voltage(params, voltage_raw)
        theory_preprocessed, _ = preprocess_raw_voltage(params, theory_raw)
        
        # align current sequence to voltage if sample frequency differs
        if voltage_preprocessed.shape[0] != current_preprocessed.shape[0]:
            current_preprocessed = align(current_preprocessed, voltage_preprocessed)
            charge_preprocessed = align(charge_preprocessed, voltage_preprocessed)
            theory_preprocessed = align(theory_preprocessed, voltage_preprocessed)

        # create input features
        profile_X1, profile_y = subsequences(current_preprocessed, voltage_preprocessed, params['n_steps'])
        profile_y = np.reshape(profile_y, (-1, 1))
        profile_X1 = profile_X1.reshape(profile_X1.shape[0], profile_X1.shape[1], 1)

        profile_X2, _ = subsequences(charge_preprocessed, voltage_preprocessed, params['n_steps'])
        profile_X2 = profile_X2.reshape(profile_X2.shape[0], profile_X2.shape[1], 1)
        
        profile_X3, _ = subsequences(theory_preprocessed, voltage_preprocessed, params['n_steps'])
        profile_X3 = profile_X3.reshape(profile_X3.shape[0], profile_X3.shape[1], 1)
        
        profile_X = np.append(profile_X1, profile_X2, axis=2)
        profile_X = np.append(profile_X, profile_X3, axis=2)

        # append for multiple profiles
        if (i == 0):
            X = profile_X
            y = profile_y
#             debug_list = charge_preprocessed
        else:
            X = np.append(X, profile_X, axis=0)
            y = np.append(y, profile_y, axis=0)
#             debug_list = np.append(debug_list, charge_preprocessed, axis=0)
        i += 1
    
#     plt.plot(debug_list)
    print('Input:', X.shape, ', Output/Label:', y.shape)    
    return X, y, scaler_volt


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
