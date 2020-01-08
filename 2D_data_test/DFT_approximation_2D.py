# In[]
import numpy as np
from scipy.fftpack import fft2,fftshift,ifft2,ifftshift
from tensorflow.keras.layers import Conv1D,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import  LearningRateScheduler,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.losses import mean_absolute_error,mean_squared_error
import tensorflow as tf
import pdb
from scipy.io import loadmat
import matplotlib.pyplot as plt 
from complex_conv import ComplexConv1D
from nn_model import min_max_model,complex_model_DFT
from signal_utils import data_read,data_prepare,integ_fftshfit,data_prepare2D
import os


main_path = './model/1_7'
if not os.path.isdir(main_path):
    os.mkdir(main_path)




number_points = 200
Ts = 4
NU_length = int(number_points/Ts)
oversample_ratio = 2
oversample_length = NU_length*oversample_ratio


input_signals = loadmat('./1_4_data/input_signal.mat')['input_signal'][:NU_length,:NU_length,1:]
input_integs = loadmat('./1_4_data/input_integ.mat')['input_integ'][:NU_length,:NU_length,:,1:] # careful about the shape of input_integs 
output_frequencys = loadmat('./1_4_data/U1_DFT.mat')['U1_DFT'][:NU_length,:NU_length,1:]


input_signals = np.transpose(input_signals,(2,0,1))
input_integs = np.transpose(input_integs,(3,0,1,2))
output_frequencys = np.transpose(output_frequencys,(2,0,1))

print(input_signals.shape)
print(input_integs.shape)
print(output_frequencys.shape)

sample_size = len(input_signals)
#print(sample_size)


#print(input_integs[0,:,:,:].reshape(2500,2))
#pdb.set_trace()


new_input_signals = np.zeros((sample_size,oversample_length,oversample_length)) + 1j*np.zeros((sample_size,oversample_length,oversample_length))
for j in range(sample_size):
    a = np.expand_dims(input_signals[j,:,:],0)
    c = input_integs[j,:,:,:]
    d = np.stack([c[:,:,0].reshape(2500,), c[:,:,1].reshape(2500,order='F')],-1)
    b = np.expand_dims(d,0)
    new_input_signals[j,:,:] = data_prepare2D(a,b,number_points)

input_signals = np.expand_dims(input_signals,-1)
output_frequencys = np.expand_dims(output_frequencys,-1)
new_input_signals = np.expand_dims(new_input_signals,-1)

print(new_input_signals.shape)
print(input_integs.shape)
print(output_frequencys.shape)

input_signals = np.concatenate([np.real(input_signals),np.imag(input_signals)],axis=-1)
new_input_signals = np.concatenate([np.real(new_input_signals),np.imag(new_input_signals)],axis=-1)
output_frequencys = np.concatenate([np.real(output_frequencys),np.imag(output_frequencys)],axis=-1)

'''
print(new_input_signals.shape)
print(input_integs.shape)
print(output_frequencys.shape)


plt.figure(1)
plt.imshow(np.abs(input_signals[0,:,:,0]))
plt.figure(2)
plt.imshow(np.abs(new_input_signals[0,:,:,0]))
'''
pdb.set_trace()

# In[]

# kernel size is 3
lr = 1e-3
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model1 = min_max_model(shape=(oversample_length,oversample_length,2),kernel_size=3)
model1.summary()
model1.compile(optimzier=Adam(lr=lr),loss=mean_squared_error)
model1.fit(new_input_signals,output_frequencys,batch_size=50,epochs=1000,verbose=1,callbacks=[lr_adapt,early_stop])
model1.save_weights(main_path+'/data_1_5_model_kernel_size_3_weights.h5')


# kernel size is 5
lr = 1e-3
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model2 = min_max_model(shape=(oversample_length,oversample_length,2),kernel_size=5)
model2.summary()
model2.compile(optimzier=Adam(lr=lr),loss=mean_squared_error)
model2.fit(new_input_signals,output_frequencys,batch_size=50,epochs=1000,verbose=1,callbacks=[lr_adapt,early_stop])
model2.save_weights(main_path+'/data_1_5_model_kernel_size_5_weights.h5')


# kernel size is 7
lr = 1e-3
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model3 = min_max_model(shape=(oversample_length,oversample_length,2),kernel_size=7)
model3.summary()
model3.compile(optimzier=Adam(lr=lr),loss=mean_squared_error)
model3.fit(new_input_signals,output_frequencys,batch_size=50,epochs=1000,verbose=1,callbacks=[lr_adapt,early_stop])
model3.save_weights(main_path+'/data_1_5_model_kernel_size_7_weights.h5')


# kernel size is 9
lr = 1e-3
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model4 = min_max_model(shape=(oversample_length,oversample_length,2),kernel_size=9)
model4.summary()
model4.compile(optimzier=Adam(lr=lr),loss=mean_squared_error)
model4.fit(new_input_signals,output_frequencys,batch_size=50,epochs=1000,verbose=1,callbacks=[lr_adapt,early_stop])
model4.save_weights(main_path+'/data_1_5_model_kernel_size_9_weights.h5')

#os.system('shutdown -s -f -t 60') # shutdown computer