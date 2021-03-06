"""
Module containing all necessary functions for the theory based thevenin model for voltage time series prediciton.
"""
import sys 
import context
import numpy as np
import matplotlib.pyplot as plt
import src.data.data_preprocessing as util
import sklearn.metrics as metrics

# -------------------------- utility functions --------------------------
def cut_profile(sequence, cutoff_time):
    cutted_seq = list()
    for i in range(len(sequence)):
        if (sequence[i,0] < cutoff_time):
            cutted_seq.append(sequence[i,1])

    return np.array(cutted_seq)

def load_profile(profile, params, cutoff_time = sys.maxsize, visualize = False):
    # ------------- load voltage -------------
    train_voltage_data = np.loadtxt('../../../data/raw/fobss_data/data/' + profile + '/cells/Slave_0_Cell_Voltages.csv', delimiter=';')
    train_voltage_data = train_voltage_data[::params['sampling_rate']]

    train_voltage_profile = train_voltage_data[:,params['battery_cell']]
    train_voltage_profile = util.smooth(train_voltage_profile, params['gauss_sigma'])

    train_voltage_time = train_voltage_data[:,0]
    train_voltage_time = np.round(train_voltage_time - train_voltage_time[0], 2)

    train_voltage = np.column_stack((train_voltage_time, train_voltage_profile))

    if (train_voltage.shape[0] > cutoff_time):
        # if profile contains unnecessary data at the end, it is cut if an appropriate cutoff_time is provided
        cut_profile(train_voltage, cutoff_time)

    # ------------- load current -------------
    train_current_data = np.loadtxt('../../../data/raw/fobss_data/data/' + profile + '/battery/Battery_Current.csv', delimiter=';')
    train_current_data = train_current_data[::params['sampling_rate']]
    train_current_data[:,1] = -train_current_data[:,1]

    train_current_profile = train_current_data[:,1]
    train_current_profile = util.smooth(train_current_profile, params['gauss_sigma'])

    train_current_time = train_current_data[:,0]
    train_current_time = np.round(train_current_time - train_current_time[0], 2)

    train_current = np.column_stack((train_current_time, train_current_profile))

    if (train_current.shape[0] > cutoff_time):
        # if profile contains unnecessary data at the end, it is cut if an appropriate cutoff_time is provided
        cut_profile(train_current, cutoff_time)
    
    if (visualize):
        fig,_ = plt.subplots(figsize=(15,10))
        # ------------- voltage -------------
        plt.subplot(2,2,1)  
        plt.plot(train_voltage_time, train_voltage_profile, color='blue')
        plt.title('Voltage')

        # ------------- current -------------
        plt.subplot(2,2,2)
        plt.plot(train_current_time, train_current_profile, color='green')
        plt.title('Current')

        # ------------- voltage gradient -------------
        volt_grad = np.gradient(train_voltage_profile) 
        plt.subplot(2,2,3)  
        plt.plot(train_voltage_time, volt_grad, color='blue')
        plt.title('Voltage Gradient')

        # ------------- current gradient -------------
        cur_grad = np.gradient(train_current_profile)
        plt.subplot(2,2,4)  
        plt.plot(train_current_time, cur_grad, color='green')
        plt.title('Current Gradient')
        plt.show()
    
    return train_current, train_voltage

# -------------------------- model parameter functions --------------------------
def identify_instant_volt_change(train_current, train_voltage):
    train_voltage_time = train_voltage[:,0]
    train_voltage_profile = train_voltage[:,1]
    train_current_time = train_current[:,0]
    train_current_profile = train_current[:,1]
    volt_grad = np.gradient(train_voltage_profile) 
    cur_grad = np.gradient(train_current_profile)
    # determine if charge or discharge profile
    if (train_voltage_profile[-1] - train_voltage_profile[0] > 0):
        # charge profile
        charge = True
    else: 
        # discharge profile
        charge = False
        
    # ------------- delta_v_0 -------------
    # find interval of instantaneous current change
    first = True
    for i in range(len(cur_grad)):
        if (charge):
            if (first and (cur_grad[i] < -0.01)):
                low_index_cur = i
                first = False
            if ((not first) and (cur_grad[i] > -0.01)):
                high_index_cur = i
                break
        else: 
            if (first and (cur_grad[i] > 0)):
                low_index_cur = i
                first = False
            if ((not first) and (cur_grad[i] < 0.01)):
                high_index_cur = i
                break

    # find interval of instantaneous voltage change
    if (charge):
        max_volt_index = np.argmax(train_voltage_profile)
    else:
        max_volt_index = np.argmin(train_voltage_profile)

    if (charge):
        max_volt_change_index = np.argmin(volt_grad)
    else:
        max_volt_change_index = np.argmax(volt_grad)


    for i in range(max_volt_index, max_volt_change_index):
        if (charge):
            if (volt_grad[i] < 0):
                low_index_volt = i
                break
        else:
            if (volt_grad[i] > 0):
                low_index_volt = i
                break

    for i in range(max_volt_change_index, train_voltage_profile.shape[0]):
        if (charge):
            if (volt_grad[i] > -0.0004):
                high_index_volt = i
                break       
        else:
            if (volt_grad[i] < 0.00015):
                high_index_volt = i
                break
                
    # ------------- delta_v_0 -------------
    # compute delta
    time_max_volt_change = train_voltage[max_volt_change_index, 0]
    print('Time of Instantaneuos Voltage Change (s):', round(time_max_volt_change, 2))
    delta_v_0 = abs(train_voltage_profile[low_index_volt] - train_voltage_profile[high_index_volt])
    print('delta_v_0 (V):', round(delta_v_0, 5))
    print('---------------------------------------------------')

    # ------------- delta_i -------------
    if (charge):
        max_cur_change = np.argmin(cur_grad)
    else:
        max_cur_change = np.argmax(cur_grad)

    print('Time of Instantaneuos Current Change (s):', round(train_current[max_cur_change, 0], 2))

    # compute delta
    delta_i = abs(train_current_profile[low_index_cur] - train_current_profile[high_index_cur])
    print('delta_i (A):', round(delta_i, 5))
    print('---------------------------------------------------')

    # ------------- R_0 -------------
    r_0 = delta_v_0 / delta_i
    print('R_0 (\u03A9):', round(r_0, 6))
    print('---------------------------------------------------')

    # ------------- Test -------------
    print('Test: delta_v = r_0 * delta_i =', round(r_0 * delta_i, 5))

    plt.plot(train_voltage_time, train_voltage_profile)
    plt.hlines(train_voltage_profile[low_index_volt], train_voltage[low_index_volt,0], train_voltage[-1,0], color='red', linestyles='dashed')
    plt.hlines(train_voltage_profile[high_index_volt], train_voltage[high_index_volt,0], train_voltage[-1,0], color='red', linestyles='dashed')
    plt.title('Instantanous Voltage Change')
    plt.show()
    
    return r_0, delta_i, max_volt_change_index

def identify_steady_state_voltage_change(train_current, train_voltage, r_0, delta_i, max_volt_change_index, params):
    train_voltage_time = train_voltage[:,0]
    train_voltage_profile = train_voltage[:,1]
    train_current_time = train_current[:,0]
    train_current_profile = train_current[:,1]
    volt_grad = np.gradient(train_voltage_profile) 
    cur_grad = np.gradient(train_current_profile)
    # determine if charge or discharge profile
    if (train_voltage_profile[-1] - train_voltage_profile[0] > 0):
        # charge profile
        charge = True
    else: 
        # discharge profile
        charge = False

    # ------------- steady state time -------------
    if (charge):
        max_decrease_index = np.argmin(volt_grad)
    else:
        max_decrease_index = np.argmax(volt_grad)

    abs_volt_grad = np.abs(volt_grad)

    # find convergence voltage 
    volt = np.round(train_voltage_profile, 5) * 100000 
    volt = volt.astype(int)
    ss_volt = np.bincount(volt).argmax() / 100000

    # find index where voltage converges
    first = True
    index = 0
    counter = 0
    for i in range(max_volt_change_index, train_voltage_profile.shape[0]):
        if (first and (np.abs(train_voltage_profile[i] - ss_volt) < 0.00001)):
            first = False
            index = i
            continue

        if ((not first) and (np.abs(train_voltage_profile[i] - ss_volt) < 0.00001)):
            counter += 1
            if (counter == params['convergence_steps']):
                # values stay the same for 100 time steps
                break

        if ((not first) and (np.abs(train_voltage_profile[i] - ss_volt) > 0.00001)):
            first = True
            index = 0

    steady_state_index = index
    
    steady_state_time = train_voltage[steady_state_index,0]
    print('Steady State Time (s):', steady_state_time)
    print('---------------------------------------------------')

    # ------------- delta_infty -------------
    if (charge):
        max_voltage = np.max(train_voltage_profile)
        max_voltage_index = np.argmax(train_voltage_profile)
    else: 
        max_voltage = np.min(train_voltage_profile)
        max_voltage_index = np.argmin(train_voltage_profile)

    steady_state_voltage = train_voltage_profile[steady_state_index]

    delta_v_infty = abs(max_voltage - steady_state_voltage)
    print('Steady State Voltage (V):', round(steady_state_voltage, 4))
    print('---------------------------------------------------')
    print('delta_v_infty (V):', round(delta_v_infty, 5))
    print('---------------------------------------------------')

    # ------------- R_1 -------------
    r_1 = (delta_v_infty - r_0 * delta_i) / delta_i
    print('R_1 (\u03A9):', round(r_1, 6))
    print('---------------------------------------------------')

    # ------------- Test -------------
    print('Test: delta_v_infty = (R_0 + R_1) * delta_i =', round((r_0 + r_1) * delta_i, 5))

    plt.plot(train_voltage_time, train_voltage_profile)
    plt.hlines(train_voltage_profile[max_voltage_index], train_voltage[max_voltage_index,0],  train_voltage[-1,0], color='red', linestyles='dashed')
    plt.hlines(train_voltage_profile[steady_state_index], train_voltage[max_decrease_index,0],  train_voltage[-1,0], color='red', linestyles='dashed')
    plt.title('Steady State Voltage Change')
    plt.show()
    
    return r_1, max_decrease_index, steady_state_time, max_voltage_index, steady_state_index

def identify_steady_state_time(train_current, train_voltage, r_1, max_decrease_index, steady_state_time, max_voltage_index, steady_state_index):
    train_voltage_time = train_voltage[:,0]
    train_voltage_profile = train_voltage[:,1]
    train_current_time = train_current[:,0]
    train_current_profile = train_current[:,1]
    volt_grad = np.gradient(train_voltage_profile) 
    cur_grad = np.gradient(train_current_profile)
    # determine if charge or discharge profile
    if (train_voltage_profile[-1] - train_voltage_profile[0] > 0):
        # charge profile
        charge = True
    else: 
        # discharge profile
        charge = False
        
    # ------------- delta_t -------------
    max_voltage_time = train_voltage[max_decrease_index,0]
    delta_t = steady_state_time - max_voltage_time 
    print('delta_t (s):', round(delta_t, 5))
    print('---------------------------------------------------')

    # C_1
    c_1 = delta_t / (4 * r_1)
    print('C_1 (F):', round(c_1, 2))
    # print(r_1 * c_1)

    # ------------- Test -------------
    print('---------------------------------------------------')
    print('Test: delta_t = 4 * R_1 * C_1 =', round(4 * r_1 * c_1, 5))
    print('Test: R_1 * C_1 =', r_1 * c_1)

    if (charge):
        upper_y = np.max(train_voltage_profile)
        lower_y = np.min(train_voltage_profile)
    else:
        upper_y = np.min(train_voltage_profile)
        lower_y = np.max(train_voltage_profile)

    plt.plot(train_voltage_time, train_voltage_profile)
    plt.vlines(train_voltage[max_voltage_index,0], upper_y, lower_y, color='red', linestyles='dashed')
    plt.vlines(train_voltage[steady_state_index,0], upper_y, lower_y, color='red', linestyles='dashed')
    plt.title('Time to Steady State')
    plt.show()
    
    return c_1

# -------------------------- SOC OCV functions --------------------------
def lin(x, x_1, y_1, x_2, y_2):
    a = (y_2 - y_1) / (x_2 - x_1)
    b = y_1 - (a * x_1)
    return a * x + b

# ------------- ocv(z) -------------
def ocv_exact(z):
    if (z < 0 or z > 1):
        return -1
    
    z = round(z, 3)
    ocv_curve = np.loadtxt('../../../data/processed/soc_ocv/ocv_curve.csv', delimiter=';')
    soc = ocv_curve[:,1]
    
    # search for best fit soc value
    delta = np.abs(z - soc)
    soc_index = np.argmin(delta)
    
    return ocv_curve[soc_index,2]

def ocv_exact_lin(z):
    ocv_curve = np.loadtxt('../../../data/processed/soc_ocv/ocv_curve.csv', delimiter=';')
    soc = ocv_curve[:,1]
    ocv = ocv_curve[:,2]
    return np.interp(z, soc, ocv)


# ------------- ocv_inverse(v) -------------
def ocv_inverse_exact(v):
    ocv_curve = np.loadtxt('../../../data/processed/soc_ocv/ocv_curve.csv', delimiter=';')
    ocv = ocv_curve[:,2]
    delta = np.abs(ocv - v)
    ocv_index = np.argmin(delta)
    z = ocv_curve[ocv_index,1]
    return round(z, 4)

def ocv_inverse_exact_lin(v, ocv_curve):
    z = np.argmin(np.abs(ocv_curve[:,1] - v))
    return ocv_curve[z,0]

# -------
def ocv_simple(z, z_1, v_1, z_2, v_2, ocv_curve):
    return lin(z * 100, z_1 * 100, v_1, z_2 * 100, v_2)

def ocv_inverse_simple(v, ocv_curve):
    z = np.argmin(np.abs(ocv_curve[:,1] - v))
    return ocv_curve[z,0]
# -------

def plot_SOC_curves(profile, params):
    # ------------- create & plot curves -------------
    steps = np.arange(0, 1, 0.001)
    percent = steps * 0.1
    volt = np.arange(3, 4, 0.001)

    ocv_curve_simple = list()
    ocv_curve_exact = list()
    ocv_curve_exact_lin = list()

    soc_curve_simple = list()
    soc_curve_exact = list()
    soc_curve_exact_lin = list()

    for i in range(len(steps)):
        ocv_curve_exact.append([steps[i], ocv_exact(steps[i])])
        ocv_curve_exact_lin.append([steps[i], ocv_exact_lin(steps[i])])

    ocv_curve_exact = np.array(ocv_curve_exact)
    ocv_curve_exact_lin = np.array(ocv_curve_exact_lin)

    # -------
    _, train_voltage = load_profile(profile, params, cutoff_time = 700)
    train_voltage_profile = train_voltage[:,1]
    v_1 = train_voltage_profile[0]
    v_2 = train_voltage_profile[-1]
    z_1 = ocv_inverse_exact_lin(v_1, ocv_curve_exact_lin)
    z_2 = ocv_inverse_exact_lin(v_2, ocv_curve_exact_lin)

    for i in range(len(steps)):
        ocv_curve_simple.append([steps[i], ocv_simple(steps[i], z_1, v_1, z_2, v_2, ocv_curve_exact_lin)])

    ocv_curve_simple = np.array(ocv_curve_simple)
    # -------

    for i in range(len(volt)):
        soc_curve_exact.append([volt[i], ocv_inverse_exact(volt[i])])
        soc_curve_exact_lin.append([volt[i], ocv_inverse_exact_lin(volt[i], ocv_curve_exact_lin)])
        soc_curve_simple.append([volt[i], ocv_inverse_simple(volt[i], ocv_curve_simple)])

    soc_curve_exact = np.array(soc_curve_exact)
    soc_curve_exact_lin = np.array(soc_curve_exact_lin)
    soc_curve_simple = np.array(soc_curve_simple)

    fig,_ = plt.subplots(figsize=(10,8))
    plt.subplot(3,3,1)
    plt.plot(ocv_curve_simple[:,0], ocv_curve_simple[:,1])
    plt.ylabel('OCV')
    plt.title('linear')
    plt.subplot(3,3,2)  
    plt.plot(ocv_curve_exact[:,0], ocv_curve_exact[:,1])
    plt.title('73 values')
    plt.subplot(3,3,3)  
    plt.plot(ocv_curve_exact_lin[:,0], ocv_curve_exact_lin[:,1])
    plt.title('73 values (interpolated)')
    plt.subplot(3,3,4)  
    plt.plot(soc_curve_simple[:,0], soc_curve_simple[:,1])
    plt.ylabel('SOC')
    plt.xlabel('OCV')
    plt.subplot(3,3,5)  
    plt.plot(soc_curve_exact[:,0], soc_curve_exact[:,1])
    plt.xlabel('OCV')
    plt.subplot(3,3,6)  
    plt.plot(soc_curve_exact[:,0], soc_curve_exact_lin[:,1])
    plt.xlabel('OCV')
    plt.show()

def get_SOC_values(profile, params):
    # ------------- create ocv curve -------------
    steps = np.arange(0, 1, 0.001)
    ocv_curve_exact_lin = list()

    for i in range(len(steps)):
        ocv_curve_exact_lin.append([steps[i], ocv_exact_lin(steps[i])])

    ocv_curve_exact_lin = np.array(ocv_curve_exact_lin)

    # ------------- determine SOC range -------------
    _, train_voltage = load_profile(profile, params, cutoff_time = 700)
    train_voltage_profile = train_voltage[:,1]
    v_1 = train_voltage_profile[0]
    v_2 = train_voltage_profile[-1]
    z_1 = ocv_inverse_exact_lin(v_1, ocv_curve_exact_lin)
    z_2 = ocv_inverse_exact_lin(v_2, ocv_curve_exact_lin)

    return z_1, v_1, z_2, v_2, ocv_curve_exact_lin

# -------------------------- z(t) --------------------------
class z_wrapper:
    def __init__(self, z_t0, q):
        self.z_k = z_t0
        self.q = q
        
    def z(self, i_k, delta_t):
        # i > 0 on discharge, i < 0 on charge
        i_k_h = i_k * (1 / 3600) # convert seconds to hours
        self.z_k = self.z_k - (i_k_h * delta_t) / self.q
        return self.z_k
    
# -------------------------- i(t) --------------------------
def i(t):
    index = round(t / 0.1)
    return current[index,1]

# -------------------------- i_r1(t) --------------------------
class i_r1_wrapper:
    def __init__(self, r_1, c_1):
        self.i_r1_k = 0
        self.r_1 = r_1
        self.c_1 = c_1
        
    def i_r1(self, i_k, delta_t):
        tau = self.r_1 * self.c_1
        exponent = - delta_t / tau
        self.i_r1_k = np.exp(exponent) * self.i_r1_k + (1 - np.exp(exponent)) * i_k
        return self.i_r1_k

# -------------------------- v(t) --------------------------
class v_wrapper:
    def __init__(self, z_t0, q, r_0, r_1, c_1):
        self.z_class = z_wrapper(z_t0, q)
        self.i_r1_class = i_r1_wrapper(r_1, c_1)
        self.r_0 = r_0
        self.r_1 = r_1
        
    def v_k(self, i_k, delta_t, z_1, v_1, z_2, v_2, ocv_curve_exact_lin):
        z_t = self.z_class.z(i_k, delta_t)
        ocv = ocv_simple(z_t, z_1, v_1, z_2, v_2, ocv_curve_exact_lin)
        i_r1_k = self.i_r1_class.i_r1(i_k, delta_t)
        return ocv - self.r_1 * i_r1_k - self.r_0 * i_k
    
# -------------------------- predict voltage --------------------------
def vis_predict(profile, r_0, r_1, c_1, params):
    # ------------- load data -------------
    test_current, test_voltage = load_profile(profile, params, cutoff_time = 700)
    test_voltage_time = test_voltage[:,0]
    test_voltage_profile = test_voltage[:,1]
    test_current_time = test_current[:,0]
    test_current_profile = test_current[:,1]
    
    z_1, v_1, z_2, v_2, ocv_curve_exact_lin = get_SOC_values(profile, params)
    
    # ------------- set parameters -------------
    v_0 = test_voltage_profile[0]
    z_t0 = ocv_inverse_simple(v_0, ocv_curve_exact_lin)
    q = 33.2 # expert knowledge

    print('------------------- Params -------------------')
    print('v_0 (V):', round(v_0, 5))
    print('z_t0 (%):', round(z_t0 * 100, 2))
    print('Q (Ah):', q)

    # ------------- predict profile -------------
    print('------------------- Results -------------------')
    v_class = v_wrapper(z_t0, q, r_0, r_1, c_1)
    vhat_profile = list()

    for i in range(test_current_profile.shape[0]):
        if (i == 0):
            d_t = 0.25 * params['sampling_rate']
        else:
            d_t = test_current_time[i] - test_current_time[i-1]

        i_k = - test_current_profile[i] # i_k is negative on charge and positive on discharge
        vhat = v_class.v_k(i_k, d_t * 2, z_1, v_1, z_2, v_2, ocv_curve_exact_lin)
        vhat_profile.append(vhat)

    vhat_profile = np.array(vhat_profile)    
    v = util.align(vhat_profile, test_voltage_profile)
    delta = np.abs(test_voltage_profile - v)

    # ------------- print results -------------
    fig, _ = plt.subplots(figsize=(7,10))
    plt.subplot(2,1,1)
    plt.plot(test_voltage_time, test_voltage_profile, label='measured', color='blue')
    plt.plot(test_voltage_time, v, label='predicted', color='red')
    plt.ylabel('voltage (V)')
    plt.title('Test Data')
    plt.legend()
    # plt.subplot(2,1,2)
    # plt.plot(test_voltage_time, delta, color='m', label='predicted - measured')
    # plt.ylabel('voltage (V)')
    # plt.xlabel('time (s)')
    # plt.title('Absolute Error')
    # plt.legend()
    plt.show()

    print('------------------- Evaluation -------------------')
    print('MSE(\u03BCV):', round(metrics.mean_squared_error(test_voltage_profile, v), 7) * 1000000)

    # save plots and predicted sequences
    MODEL_ID = str(np.random.randint(10000))
    print('Saved plot to:', '../../../reports/figures/theory_baseline-' + str(MODEL_ID) + '-' + profile + '-test_profile.png')
    fig.savefig('../../../reports/figures/theory_baseline-' + str(MODEL_ID) + '-' + profile + '-test_profile.png')
    return v
    
def predict(profile, r_0, r_1, c_1, params):
    # ------------- load data -------------
    test_current, test_voltage = load_profile(profile, params, cutoff_time = 700)
    test_voltage_time = test_voltage[:,0]
    test_voltage_profile = test_voltage[:,1]
    test_current_time = test_current[:,0]
    test_current_profile = test_current[:,1]
    
    z_1, v_1, z_2, v_2, ocv_curve_exact_lin = get_SOC_values(profile, params)
    
    # ------------- set parameters -------------
    v_0 = test_voltage_profile[0]
    z_t0 = ocv_inverse_simple(v_0, ocv_curve_exact_lin)
    q = 33.2 # expert knowledge

    # ------------- predict profile -------------
    v_class = v_wrapper(z_t0, q, r_0, r_1, c_1)
    vhat_profile = list()

    for i in range(test_current_profile.shape[0]):
        if (i == 0):
            d_t = 0.25 * params['sampling_rate']
        else:
            d_t = test_current_time[i] - test_current_time[i-1]

        i_k = - test_current_profile[i] # i_k is negative on charge and positive on discharge
        vhat = v_class.v_k(i_k, d_t * 2, z_1, v_1, z_2, v_2, ocv_curve_exact_lin)
        vhat_profile.append(vhat)

    vhat_profile = np.array(vhat_profile)    
    v = util.align(vhat_profile, test_voltage_profile)
    return v
