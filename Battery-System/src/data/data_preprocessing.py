"""
Module containing different utility functions used for preprocessing time series data.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage 
import matplotlib.pyplot as plt

import src.models.thevenin_model as thevenin

# ---------------------------------------------- Load Data -------------------------------------------------------
def load_current_raw_data(profile):
    """Loads the current raw data.

    This method will extract raw data provided by the battery current sensor.

    Args:
        profile (str):
            The FOBSS profile from which the data should be loaded
        
    Returns:
        A numpy array containing the requested raw data.
    """
    current_data = np.loadtxt('../../../data/raw/fobss_data/data/' + profile + '/battery/Battery_Current.csv', delimiter=';') # TODO: propably a terrible idea
    current_data = -current_data[:,1] # only the first column includes necessary information
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
        sequence_X (numpy.ndarray): 
            The first sequence which gets converted into multiple subarrays of length: n_steps
            
        sequence_y (numpy.ndarray): 
            The second sequence, each n_steps'th element will be part of the output array
            
        n_steps (int): 
            The amount of time steps used as an input into the LSTM for prediction

    Returns:
        A tuple of 2 numpy arrays in the required format.

        X.shape = (sequence_X.shape[0] - n_steps, n_steps)
        y.shape = (sequence_y.shape[0] - n_steps, 1)

    Raises:
        Exception: If n_steps exceeds the length of sequence_X no subsequences can be created.
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

    In this context this means subsampling the first array so that it afterwards has the 
    same size as the second array.
    
    Args: 
        sequence_1 (numpy.ndarray): 
            The arrray to be aligned (the one with bigger size)
            
        sequence_2 (numpy.ndarray): 
            The array to be aligned to

    Returns:
        The aligned array
        
    Raises:
        Exception: If the array which is being aligned is of smaller size than the one 
        which it is supposed to be aligned to.
    """
    if len(sequence_1) < len(sequence_2):
        raise Exception('data_preprocessing.align: missmatch of sequence lengths')
    
    sample_ratio = sequence_1.shape[0] / sequence_2.shape[0]

    aligned_sequence = list()
    for i in range(len(sequence_2)):
        aligned_sequence.append(sequence_1[int(np.round(i * sample_ratio))])

    aligned_sequence = np.array(aligned_sequence)
    return aligned_sequence


def preprocess_raw_current(params, sequence):
    """Preprocesses the raw current by subsamling, smoothing and scaling the data.
    Args:
        params (dict): 
            Dictionary containing the keys 'd_sample', 'gauss_sigma', 'feature_range_cur_low',
            'feature_range_cur_high', 'boundary_cur_low', 'boundary_cur_high'
            
        sequence (numpy.ndarray): 
            The sequence to be preprocessed
        
    Returns:
        A tuple of 2 values. The preprocessed sequence and the scaler object 
        (used for retransforming after training).
    """
    sequence = subsample(sequence, params['d_sample'])
    sequence = smooth(sequence, params['gauss_sigma'])
    sequence = np.reshape(sequence, (-1, 1))
    
    scaler = MinMaxScaler(feature_range=(params['feature_range_cur_low'], params['feature_range_cur_high']))
    
    boundaries = [params['boundary_cur_low'], params['boundary_cur_high']]
    boundaries = np.reshape(boundaries, (-1, 1))
    scaler.fit(boundaries)
    
    sequence = scaler.transform(sequence)
    return sequence, scaler


def preprocess_raw_charge(params, sequence):
    """Preprocesses the raw charge by subsamling, smoothing and scaling the data.
    Args:
        params (dict): 
            Dictionary containing the keys 'd_sample', 'gauss_sigma', 'feature_range_charge_low',
            'feature_range_charge_high', 'boundary_charge_low', 'boundary_charge_high'
            
        sequence (numpy.ndarray): 
            The sequence to be preprocessed
        
    Returns:
        A tuple of 2 values. The preprocessed sequence and the scaler object 
        (used for retransforming after training).
    """
    sequence = np.reshape(sequence, (-1, 1))
    sequence = subsample(sequence, params['d_sample'])

    scaler = MinMaxScaler(feature_range=(params['feature_range_charge_low'], params['feature_range_charge_high']))
    
    boundaries = [params['boundary_charge_low'], params['boundary_charge_high']]
    boundaries = np.reshape(boundaries, (-1, 1))
    scaler.fit(boundaries)
    
    sequence = scaler.transform(sequence)
    return sequence, scaler


def preprocess_raw_voltage(params, sequence):
    """Preprocesses the raw voltage by subsamling, smoothing and scaling the data.
    Args:
        params (dict): 
            Dictionary containing the keys 'd_sample', 'gauss_sigma', 'feature_range_volt_low',
            'feature_range_volt_high', 'boundary_voltage_low', 'boundary_voltage_high'
            
        sequence (numpy.ndarray): 
            The sequence to be preprocessed
        
    Returns:
        A tuple of 2 values. The preprocessed sequence and the scaler object 
        (used for retransforming after training).
    """
    sequence = subsample(sequence, params['d_sample'])
    sequence = smooth(sequence, params['gauss_sigma'])
    sequence = np.reshape(sequence, (-1, 1))
    
    scaler = MinMaxScaler(feature_range=(params['feature_range_volt_low'], params['feature_range_volt_high']))
    
    boundaries = [params['boundary_voltage_low'], params['boundary_voltage_high']]
    boundaries = np.reshape(boundaries, (-1, 1))
    scaler.fit(boundaries)
    
    sequence = scaler.transform(sequence)
    
    return sequence, scaler


def preprocess_raw_thevenin(params, sequence):
    """Preprocesses the raw thevenin voltage by smoothing and scaling the data.
    Args:
        params (dict): 
            Dictionary containing the keys 'd_sample', 'gauss_sigma', 'feature_range_volt_low',
            'feature_range_volt_high', 'boundary_voltage_low', 'boundary_voltage_high'
            
        sequence (numpy.ndarray): 
            The sequence to be preprocessed
        
    Returns:
        A tuple of 2 values. The preprocessed sequence and the scaler object 
        (used for retransforming after training).
    """
    sequence = smooth(sequence, params['gauss_sigma'])
    sequence = np.reshape(sequence, (-1, 1))
    
    scaler = MinMaxScaler(feature_range=(params['feature_range_volt_low'], params['feature_range_volt_high']))
    
    boundaries = [params['boundary_voltage_low'], params['boundary_voltage_high']]
    boundaries = np.reshape(boundaries, (-1, 1))
    scaler.fit(boundaries)
    
    sequence = scaler.transform(sequence)
    
    return sequence, scaler

# -------------------------------------- Prediction Thevenin Model -----------------------------------------------
def predict_thevenin(params, profile):
    """Predicts the voltage using the Thevenin model.
    Args:
        params (dict): 
            Dictionary containing the key 'theory_model'
            
        profile (str): 
            The profile to be predicted
        
    Returns:
        The voltage prediction of the provided profile using the parametrized Thevenin Model.
    """
    thevenin_params = np.load('../../../models/T/theory_baseline-' + str(params['theory_model']) + '-parameters.npy', allow_pickle=True)
    thevenin_params = thevenin_params.item()

    thevenin_hyperparams = np.load('../../../models/T/theory_baseline-' + str(params['theory_model']) + '-hyperparameters.npy', allow_pickle=True)
    thevenin_hyperparams = thevenin_hyperparams.item()

    update_params = {
        'd_sample': params['d_sample']
    }
    thevenin_hyperparams.update(update_params)
    
    theory_raw = thevenin.predict(profile, thevenin_params['r_0'], thevenin_params['r_1'], thevenin_params['c_1'], thevenin_hyperparams)
    
    return theory_raw

# ---------------------------------------- Prepare Network Input -------------------------------------------------

def prepare_thevenin_pred(params, profiles):
    """Predicts multiple voltage profiles using the Thevenin Model.
    Args:
        params (dict): 
            Dictionary containing the key 'theory_model'
            
        profiles (list): 
            The profiles to be predicted
        
    Returns:
        The voltage prediction of the provided profiles using the parametrized Thevenin Model.
    """
    i = 0
    for profile in profiles:

        if (i == 0):
            uhat = predict_thevenin(params, profile)
        else:
            uhat = np.append(uhat, predict_thevenin(params, profile), axis=0)
        i += 1

    uhat, _ = preprocess_raw_voltage(params, uhat)
    return uhat

# ---------------------------------------- Data Baseline -------------------------------------------------
def prepare_current_input(params, profiles, slave, cell):
    """Prepares the current data to be suitable for the network.

    Args:
        params (dict): 
            Dictionary containing the appropriate keys for preprocessing
            
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
        voltage_raw = load_voltage_raw_data(profile, slave, cell)
        
        # preprocess data
        current_preprocessed, _ = preprocess_raw_current(params, current_raw)    
        voltage_preprocessed, scaler_volt = preprocess_raw_voltage(params, voltage_raw)

        # align current sequence to voltage if sample frequency differs
        if voltage_preprocessed.shape[0] != current_preprocessed.shape[0]:
            current_preprocessed = align(current_preprocessed, voltage_preprocessed)

        # create input features
        profile_X, profile_y = subsequences(current_preprocessed, voltage_preprocessed, params['n_steps'])
        profile_y = np.reshape(profile_y, (-1, 1))
        profile_X = profile_X.reshape(profile_X.shape[0], profile_X.shape[1], 1)

        # append for multiple profiles
        if (i == 0):
            X = profile_X
            y = profile_y
        else:
            X = np.append(X, profile_X, axis=0)
            y = np.append(y, profile_y, axis=0)
        i += 1
    
    print('Input:', X.shape, ' Output/Label:', y.shape)
    return X, y, scaler_volt

# ---------------------------------------- Feature Engineering -------------------------------------------------
def prepare_feature_engineering_input(params, profiles, slave, cell):
    """Prepares the feature engineering input data to be suitable for the network.

    Args:
        params (dict): 
            Dictionary containing the appropriate keys for preprocessing
            
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
        
        charge_raw = []
        q_t = 0
        for j in range(len(current_raw)):
            q_t += current_raw[j] * (params['d_t'] * params['d_sample'])  / 3600
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
    
    print('Input:', X.shape, ' Output/Label:', y.shape)
    return X, y, scaler_volt

# --------------------------------------- Intermediate Variables ------------------------------------------------
def prepare_intermediate_input(params, profiles, slave, cell):
    """Prepares the intermediate variables input data to be suitable for the network.

    Args:
        params (dict): 
            Dictionary containing the appropriate keys for preprocessing
            
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
        
        charge_raw = []
        q_t = 0
        for j in range(len(current_raw)):
            q_t += current_raw[j] * (params['d_t'] * params['d_sample']) / 3600
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
        profile_y = np.append(profile_y, current_preprocessed, axis=1) 
        
        # append for multiple profiles
        if (i == 0):
            X = profile_X
            y = profile_y
        else:
            X = np.append(X, profile_X, axis=0)
            y = np.append(y, profile_y, axis=0)
        i += 1
    
    print('Input:', X.shape, ' Output/Label:', y.shape)
    return X, y, scaler_volt

# ---------------------------------------- Initialization -------------------------------------------------
def prepare_pretraining_input(params, profiles, slave, cell):
    """Prepares the pretraining input data to be suitable for the network.

    Args:
        params (dict): 
            Dictionary containing the appropriate keys for preprocessing
            
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
        
        charge_raw = []
        q_t = 0
        for j in range(len(current_raw)):
            q_t += current_raw[j] * (params['d_t'] * params['d_sample'])  / 3600
            charge_raw.append(q_t)
        
        # predict on theory model
        theory_raw = predict_thevenin(params, profile)
        
        # preprocess data
        current_preprocessed, _ = preprocess_raw_current(params, current_raw)    
        charge_preprocessed, _ = preprocess_raw_charge(params, charge_raw)
        theory_preprocessed, scaler_volt = preprocess_raw_voltage(params, theory_raw)
        
        # align current sequence to voltage if sample frequency differs
        if theory_preprocessed.shape[0] != current_preprocessed.shape[0]:
            current_preprocessed = align(current_preprocessed, theory_preprocessed)
            charge_preprocessed = align(charge_preprocessed, theory_preprocessed)

        # create input features
        profile_X1, profile_y = subsequences(current_preprocessed, theory_preprocessed, params['n_steps'])
        profile_y = np.reshape(profile_y, (-1, 1))
        profile_X1 = profile_X1.reshape(profile_X1.shape[0], profile_X1.shape[1], 1)

        profile_X2, _ = subsequences(charge_preprocessed, theory_preprocessed, params['n_steps'])
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
    
    print('Input:', X.shape, ' Output/Label:', y.shape)
    return X, y, scaler_volt

# ---------------------------------------- Model Design -------------------------------------------------
def prepare_thevenin_input(params, profiles, slave, cell):
    """Prepares the model design input data to be suitable for the network.

    Args:
        params (dict): 
            Dictionary containing the appropriate keys for preprocessing
            
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
        voltage_raw = load_voltage_raw_data(profile, slave, cell)
        
        charge_raw = []
        q_t = 0
        for j in range(len(current_raw)):
            q_t += current_raw[j] * (params['d_t'] * params['d_sample'])  / 3600
            charge_raw.append(q_t)
        
        
        # load SOC
        _, _, _, _, ocv_curve_exact_lin = thevenin.get_SOC_values(profile, params)
        v_0 = voltage_raw[0]
        z_t0 = thevenin.ocv_inverse_exact_lin(v_0, ocv_curve_exact_lin)
        q = 33.2 # expert knowledge
        z_class = thevenin.z_wrapper(z_t0, q)
        
        # get OCV
        ocv = []
        for j in range(len(current_raw)):
            i_k = current_raw[j]
            z_k = z_class.z(-i_k, params['d_t']) # i > 0 on discharge, i < 0 on charge
            ocv.append(thevenin.ocv_exact_lin(z_k))

        # preprocess data
        current_preprocessed, _ = preprocess_raw_current(params, current_raw)    
        ocv_preprocessed, _ = preprocess_raw_voltage(params, ocv)
        charge_preprocessed, _ = preprocess_raw_charge(params, charge_raw)
        voltage_preprocessed, scaler_volt = preprocess_raw_voltage(params, voltage_raw)

        # align current sequence to voltage if sample frequency differs
        if voltage_preprocessed.shape[0] != current_preprocessed.shape[0]:
            current_preprocessed = align(current_preprocessed, voltage_preprocessed)
            ocv_preprocessed = align(ocv_preprocessed, voltage_preprocessed)
            charge_preprocessed = align(charge_preprocessed, voltage_preprocessed)

        # create input features
        profile_X1, profile_y = subsequences(current_preprocessed, voltage_preprocessed, params['n_steps'])
        profile_y = np.reshape(profile_y, (-1, 1))
        profile_X1 = profile_X1.reshape(profile_X1.shape[0], profile_X1.shape[1], 1)
        
        profile_X2, _ = subsequences(ocv_preprocessed, voltage_preprocessed, params['n_steps'])
        profile_X2 = profile_X2.reshape(profile_X2.shape[0], profile_X2.shape[1], 1)
        
        profile_X3, _ = subsequences(charge_preprocessed, voltage_preprocessed, params['n_steps'])
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
    
    print('Input:', X.shape, ' Output/Label:', y.shape)
    return X, y, scaler_volt


def prepare_intermediate_soc(params, profiles, slave, cell):
    """Prepares the SOC as a non-squential input variable to the network.

    Args:
        params (dict): 
            Dictionary containing the appropriate keys for preprocessing
            
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
        voltage_raw = load_voltage_raw_data(profile, slave, cell)
        
        # compute current SOC at start of profile
        soc = thevenin.ocv_inverse_exact(voltage_raw[0])
        initial_soc = np.repeat(soc, len(voltage_raw))

        # preprocess data
        voltage_preprocessed, scaler_volt = preprocess_raw_voltage(params, voltage_raw)
        
        # align current sequence to voltage if sample frequency differs
        if voltage_preprocessed.shape[0] != initial_soc.shape[0]:
            initial_soc = align(initial_soc, voltage_preprocessed)

        # create input feature sequence
        sequence_X = np.append(np.repeat(initial_soc[0], params['n_steps'] - 1), initial_soc)

        profile_X = list()
        for j in range(len(sequence_X)):
            end_ix = j + params['n_steps']

            if end_ix > len(sequence_X):
                break

            profile_X.append(sequence_X[j])
            
        profile_X = np.array(profile_X)
        profile_X = np.reshape(profile_X, (-1, 1))        
        
        # append for multiple profiles
        if (i == 0):
            X = profile_X
        else:
            X = np.append(X, profile_X, axis=0)
        i += 1
    
    print('Input:', X.shape) 
    return X

def prepare_intermediate_volt(params, profiles, slave, cell):
    """Prepares the initial voltage as a non-squential input variable to the network.

    Args:
        params (dict): 
            Dictionary containing the appropriate keys for preprocessing
            
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
        voltage_raw = load_voltage_raw_data(profile, slave, cell)
        initial_voltage = np.repeat(voltage_raw[0], len(voltage_raw))

        # compute current SOC at start of profile
        soc = thevenin.ocv_inverse_exact(voltage_raw[0])
        initial_soc = np.repeat(soc, len(voltage_raw))

        # preprocess data
        voltage_preprocessed, scaler_volt = preprocess_raw_voltage(params, voltage_raw)
        intial_voltage_preprocessed, _ = preprocess_raw_voltage(params, initial_voltage)

        # align current sequence to voltage if sample frequency differs
        if voltage_preprocessed.shape[0] != intial_voltage_preprocessed.shape[0]:
            intial_voltage_preprocessed = align(intial_voltage_preprocessed, voltage_preprocessed)

        # create input feature sequence
        sequence_X = np.append(np.repeat(intial_voltage_preprocessed[0], params['n_steps'] - 1), intial_voltage_preprocessed)

        profile_X = list()
        for j in range(len(sequence_X)):
            end_ix = j + params['n_steps']

            if end_ix > len(sequence_X):
                break

            profile_X.append(sequence_X[j])
            
        profile_X = np.array(profile_X)
        profile_X = np.reshape(profile_X, (-1, 1))        
        # ----
        
        # append for multiple profiles
        if (i == 0):
            X = profile_X
        else:
            X = np.append(X, profile_X, axis=0)
        i += 1
        
    return X

# ---------------------------------------- Hybrid Model -------------------------------------------------
def prepare_hybrid_input(params, profiles, slave, cell):
    """Prepares the hybrid model input data to be suitable for the network.

    Args:
        params (dict): 
            Dictionary containing the appropriate keys for preprocessing
            
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
        voltage_raw = load_voltage_raw_data(profile, slave, cell)
        
        # predict on theory model
        theory_raw = predict_thevenin(params, profile)
        
        charge_raw = []
        q_t = 0
        for j in range(len(current_raw)):
            q_t += current_raw[j] * (params['d_t'] * params['d_sample'])  / 3600
            charge_raw.append(q_t)
            
        # preprocess data
        current_preprocessed, _ = preprocess_raw_current(params, current_raw)    
        voltage_preprocessed, scaler_volt = preprocess_raw_voltage(params, voltage_raw)
        theory_preprocessed, _ = preprocess_raw_thevenin(params, theory_raw)
        charge_preprocessed, _ = preprocess_raw_charge(params, charge_raw)
        
        # align current sequence to voltage if sample frequency differs
        if voltage_preprocessed.shape[0] != current_preprocessed.shape[0]:
            current_preprocessed = align(current_preprocessed, voltage_preprocessed)
            charge_preprocessed = align(charge_preprocessed, voltage_preprocessed)

        # create input features
        profile_X1, profile_y = subsequences(current_preprocessed, voltage_preprocessed, params['n_steps'])
        profile_y = np.reshape(profile_y, (-1, 1))
        profile_X1 = profile_X1.reshape(profile_X1.shape[0], profile_X1.shape[1], 1)

        profile_X2, _ = subsequences(theory_preprocessed, voltage_preprocessed, params['n_steps'])
        profile_X2 = profile_X2.reshape(profile_X2.shape[0], profile_X2.shape[1], 1)
        
        profile_X3, _ = subsequences(charge_preprocessed, voltage_preprocessed, params['n_steps'])
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
    
    print('Input:', X.shape, ' Output/Label:', y.shape)    
    return X, y, scaler_volt

# ---------------------------------------- Residual Learning -------------------------------------------------
def prepare_residual_input(params, profiles, slave, cell):
    """Prepares the residual learning input data to be suitable for the network.

    Args:
        params (dict): 
            Dictionary containing the appropriate keys for preprocessing
            
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
        
        voltage_raw = load_voltage_raw_data(profile, slave, cell)
        theory_raw = predict_thevenin(params, profile)
        
        charge_raw = []
        q_t = 0
        for j in range(len(current_raw)):
            q_t += current_raw[j] * (params['d_t'] * params['d_sample'])  / 3600
            charge_raw.append(q_t)      
        
        # preprocess data
        current_preprocessed, _ = preprocess_raw_current(params, current_raw)    
        voltage_preprocessed, scaler_volt = preprocess_raw_voltage(params, voltage_raw)
        theory_preprocessed, _ = preprocess_raw_thevenin(params, theory_raw)
        charge_preprocessed, _ = preprocess_raw_charge(params, charge_raw)
        
        # repare residual data
        residual_preprocessed = voltage_preprocessed - theory_preprocessed

        # align current sequence to voltage if sample frequency differs
        if residual_preprocessed.shape[0] != current_preprocessed.shape[0]:
            current_preprocessed = align(current_preprocessed, residual_preprocessed)
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
            u = theory_preprocessed
        else:
            X = np.append(X, profile_X, axis=0)
            y = np.append(y, profile_y, axis=0)
            u = np.append(u, theory_preprocessed, axis=0)
        i += 1
    
    print('Input:', X.shape, ' Output/Label:', y.shape)    
    return X, y, u, scaler_volt
