import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage 

""" Creates subsequences of the original sequence to fit LSTM structure
 
Args:
    sequence_1: the first sequence which gets converted into multiple subarrays of length: n_steps
    sequence_2: the second sequence, each n_steps'th element will be part of the output array
    n_steps: the amount of time steps used as an input into the LSTM for prediction

Returns:
    A tuple of 2 numpy arrays in the required format
    
    X.shape = (X.shape[0] - n_steps, n_steps)
    y.shape = (X.shape[0] - n_steps, 1)

"""
def subsequences(sequence_X, sequence_y, n_steps):
    if n_steps > len(sequence_X):
        raise Exception('subsequences: n_steps should not exceed the sequence length')
    
    sequence_X = np.append(np.repeat(sequence_X[0], n_steps - 1), sequence_X)
    
    X, y = list(), list()
    for i in range(len(sequence_X)):
        end_ix = i + n_steps

        if end_ix > len(sequence_X):
            break

        X.append(sequence_X[i:end_ix])
        y.append(sequence_y[i])
        
    return np.array(X), np.array(y)


""" Subsample array to decrease the amount of data

Args:
    sequence: the input array to be subsampled
    d_sample: sample frequency, meaning every d_sample'th element will be part of the output
    
Returns:
    The subsampled array

"""
def subsample(sequence, d_sample):
    return sequence[::d_sample]


""" Smooth array to decrease measurement noise

Args: 
    sequence: the input array to be smoothed
    sigma: parameter for the gauss filtering

Returns:
    The smoothed array
"""
def smooth(sequence, sigma):
    return ndimage.filters.gaussian_filter(sequence, sigma)


""" Aligns two sequences

    In this context this means subsampling the first array so that it afterwards has the same size as the second array
    
Args: 
    sequence_1: arrray to be aligned
    sequence_2: array to be aligned to
    
Returns:
    The algined array
"""
def align(sequence_1, sequence_2):
    if len(sequence_1) < len(sequence_2):
        raise Exception('align: missmatch of sequence lengths')
    
    sample_ratio = sequence_1.shape[0] / sequence_2.shape[0]

    aligned_sequence = list()
    for i in range(len(sequence_2)):
        aligned_sequence.append(sequence_1[int(np.round(i * sample_ratio))])

    aligned_sequence = np.array(aligned_sequence)
    
    return aligned_sequence


""" Prepares the data for input into the LSTM

    Preparation incudes:
    subsampling, smoothing, aligning differnt sized sequences and reshaping the sequence to the requested format
    
Args:
    input_sequence: the input feature sequence
    label_sequence: the output/groud truth sequence
    aligned: indicates if input and label sequence are of equal size or need alignment
    d_sample: sample frequency
    n_steps: the amount of time steps used as an input into the LSTM for prediction
    sigma: parameter for the data smoothing

Returns:
    A tuple of 3 values. The prepared input sequence X, the output sequence of labels y and the scaler component for y. 
    This is needed afterwards to scale the output back to the original value range
"""
def prepare_data(input_sequence, label_sequence, aligned, d_sample, n_steps, sigma):
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

def prepare_data_unscaled(input_sequence, label_sequence, aligned, d_sample, n_steps, sigma):
    # align data if not of equal size
    if not aligned:        
        input_sequence = align(input_sequence, label_sequence)

    # subsample and smooth data 
    input_sequence_ = subsample(input_sequence, d_sample)
    input_sequence_ = smooth(input_sequence_, gauss_sigma)
    
    label_sequence_ = subsample(label_sequence, d_sample)
    label_sequence_ = smooth(label_sequence_, gauss_sigma)

    # convert into X and y sequences
    X, y = subsequences(input_sequence_, label_sequence_, n_steps)
    y = np.reshape(y, (-1, 1))

    # reshape into correct format
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y