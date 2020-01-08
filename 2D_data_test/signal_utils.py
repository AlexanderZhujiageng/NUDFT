# In[]
import numpy as np 
from scipy.io import loadmat
from scipy.fftpack import fft,ifft,fftshift,ifftshift,fft2,ifft2
import time
import pdb 

def data_read(input_signal_file='./data/input_signal.mat',
              U1_inputs_file='./data/U1_input.mat',
              input_integ_file='./data/input_integ.mat',
              true_output_signal_file='./data/true_output_signal.mat'):
    input_signals = loadmat(input_signal_file)['input_signal']
    input_integs = loadmat(input_integ_file)['input_integ'] # integ in the original gridding 
    U1_inputs = loadmat(U1_inputs_file)['U1'] # input data in frequency domain
    true_output_signals = loadmat(true_output_signal_file)['true_output_signal'] # true result y
    
    input_integs = input_integs[:250,:]
    U1_inputs = U1_inputs[:250,:]
    
    print('the integs of input signal shape is:')
    print(input_integs.shape)

    print('the input signal shape is:')
    print(U1_inputs.shape)
    print(type(U1_inputs))

    print('the output signal shape is:')
    print(true_output_signals.shape)
    print(type(true_output_signals))

    return input_signals,input_integs,U1_inputs,true_output_signals


def integ_fftshfit(input_integs,number_points):
    '''
    because we have to do the fftshift before doing the fft to calculate the frequency response of input signal
    and this will cause the integ different for non-uniform sampling, so the new integs should be calculated
    '''
    if np.mod(number_points,2) == 0: 
        input_integs = input_integs
        new_input_integs = input_integs + np.floor(number_points/2) 
        new_input_integs = np.where(new_input_integs>number_points, new_input_integs-number_points,new_input_integs)
        return new_input_integs
    else:
        input_integs = input_integs
        new_input_integs = input_integs + np.floor(number_points/2)
        new_input_integs = np.where(new_input_integs>=number_points,new_input_integs-number_points+1,new_input_integs)
        return new_input_integs



def data_prepare(input_signals,input_integs,number_points,oversampling_ratio=2):
    '''
    This function is used to generate a over-sampling signal and place the value on the new gridding indexes
    input_signals: non uniform sampling signal on the input plane
    input_integs: non uniform sampling location in the original gridding to be sure that in matlab start from 1
    oversampling_ratio: the ratio to over-sample non-uniform signal
    number_points: the number of points in the original signal
    '''
    input_signals = input_signals
    #print(input_integs)
    
    input_integs = input_integs - 1
    input_integs = np.where(input_integs<0,0,input_integs)
    #print(input_integs)
    signal_length = input_signals.shape[-1]
    sample_size = len(input_signals)
    Nr = oversampling_ratio*signal_length

    interval_for_Nr = number_points/Nr
    integs_on_new_gridding = input_integs/interval_for_Nr



    new_input_signals = np.zeros((sample_size,Nr+1)) + 1j*np.zeros((sample_size,Nr+1))

    start_time = time.time()
    ceil_integs_on_new_gridding = np.ceil(integs_on_new_gridding).astype(np.int)
    floor_integs_on_new_gridding = np.floor(integs_on_new_gridding).astype(np.int)
    print('the np.ceil() and np.floor() takes %s second'%(time.time()-start_time))
  

    start_time = time.time()
    floor_weights = ceil_integs_on_new_gridding - integs_on_new_gridding
    ceil_weights = integs_on_new_gridding - floor_integs_on_new_gridding
    # because ceil and floor of integ result is itself,the ceil weight is 0, make it to one or extract this to make two signal  
    ceil_weights = np.where(ceil_weights==0,1,ceil_weights) 
    print('the np.where() takes %s second'%(time.time()-start_time))

    
    start_time = time.time()
    floor_values = np.multiply(input_signals,floor_weights)
    ceil_values = np.multiply(input_signals,ceil_weights)
    print('the np.multiply() takes %s second'%(time.time()-start_time))
    

    start_time = time.time()
    new_input_signals[:,ceil_integs_on_new_gridding] += ceil_values
    new_input_signals[:,floor_integs_on_new_gridding] += floor_values
    print('the add takes %s second'%(time.time()-start_time))

    
    return new_input_signals[:,:500]


def data_prepare2D(input_signals,input_integs,number_points,oversampling_ratio=2):
    '''
    generate 2D over-sampling signal and place the value on new gridding mesh.
    input_signals: non uniform 2D signal whose shape is (number_samples * N * N)
    input_integs: non uniform 2D signal location in original gridding and the shape is (number_samples * N^2 *2 )
    number_points: number of points in original signal in one dimension 
    '''
    input_signals = input_signals
    input_integs = input_integs
    input_integs = np.where(input_integs<0,0,input_integs)
    signal_H = input_signals.shape[-1]
    signal_W = input_signals.shape[-2]
    sample_size = len(input_signals)
    Nr = oversampling_ratio * signal_H 

    interval_for_Nr = number_points/Nr
    integs_on_new_gridding = input_integs / interval_for_Nr
    new_input_signals = np.zeros((sample_size,Nr+1,Nr+1)) + 1j*np.zeros((sample_size,Nr+1,Nr+1))

    # find the left_up (floor,floor), left_down(floor,ceil), right_up(ceil,floor), right_down(ceil,ceil) location value and their location
    all_ceil = np.ceil(integs_on_new_gridding).astype(np.int64) # left_up
    all_floor= np.floor(integs_on_new_gridding).astype(np.int64) # right down
    

    left_up = all_floor
    left_up_distance = 1-(integs_on_new_gridding - left_up)
    left_up_distance = np.where(left_up_distance==1,0,left_up_distance)
    left_up_weights = np.multiply(left_up_distance[:,:,0],left_up_distance[:,:,1]) 
    left_up_values = np.multiply(input_signals,left_up_weights.reshape(sample_size,signal_H,signal_W))

    right_down = all_ceil
    right_down_distance = 1-(right_down - integs_on_new_gridding)
    #right_down_distance = np.where(right_down_distance==1,0,right_down_distance)
    right_down_weights = np.multiply(right_down_distance[:,:,0], right_down_distance[:,:,1])
    right_down_weights = np.where(right_down_weights==1.,0,right_down_weights)
    right_down_values = np.multiply(input_signals,right_down_weights.reshape(sample_size,signal_H,signal_W))

    left_down = np.zeros((sample_size,signal_H*signal_W,2)).astype(np.int64)
    left_down[:,:,0] = all_floor[:,:,0]
    left_down[:,:,1] = all_ceil[:,:,1]
    left_down_distance = 1-np.abs(left_down - integs_on_new_gridding)
    left_down_distance = np.where(left_down_distance==1,0,left_down_distance)
    left_down_weights = np.multiply(left_down_distance[:,:,0], left_down_distance[:,:,1])
    left_down_values = np.multiply(input_signals,left_down_weights.reshape(sample_size,signal_H,signal_W))


    right_up = np.zeros((sample_size,signal_H*signal_W,2)).astype(np.int64)
    right_up[:,:,0] = all_ceil[:,:,0]
    right_up[:,:,1] = all_floor[:,:,1]
    right_up_distance = 1-np.abs(right_up - integs_on_new_gridding)
    right_up_distance = np.where(left_up_distance==1,0,right_up_distance)
    right_up_weights = np.multiply(right_up_distance[:,:,0], right_up_distance[:,:,1])
    right_up_values = np.multiply(input_signals,right_up_weights.reshape(sample_size,signal_H,signal_W))

    

    new_input_signals[:,left_up[:,:,0],left_up[:,:,1]] += left_up_values.reshape(sample_size,signal_H*signal_W)

   

    new_input_signals[:,right_up[:,:,0],right_up[:,:,1]] += right_up_values.reshape(sample_size,signal_H*signal_W)
  
    
    new_input_signals[:,left_down[:,:,0],left_down[:,:,1]] += left_down_values.reshape(sample_size,signal_H*signal_W)
 
    
    new_input_signals[:,right_down[:,:,0],right_down[:,:,1]] += right_down_values.reshape(sample_size,signal_H*signal_W)
 
    
    return new_input_signals[:,:Nr,:Nr]

'''
# data_prepare2D test code below   
input_signal = np.expand_dims(np.array([[2.,3.],[4.,5.]]),0)
print(input_signal.shape)


input_integ = np.expand_dims(np.array([[0,0],[2,2],[6.25,5],[7,7]]),0)
print(input_integ.shape)
number_points = 10


#In[]

new_signals = data_prepare2D(input_signal,input_integ,number_points)
print(new_signals)
print(new_signals[:,2,3])

#data_prepare test code below

input_signal = np.expand_dims(np.array([2.,3.,4.,5.]),0)
input_signals = np.concatenate([input_signal,input_signal],axis=0)

input_integ = np.expand_dims(np.array([0, 2, 5, 7]),0)
input_integs = np.concatenate([input_integ,input_integ],axis=0)
number_points = 10

new_signals = data_prepare(input_signals,input_integs,number_points)
print(np.abs(new_signals))
'''

# %%
