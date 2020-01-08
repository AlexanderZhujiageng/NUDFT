import numpy as np 
from scipy.fftpack import fft,ifft,fftshift,ifftshift
import tensorflow.keras 
import tensorflow.keras.backend as K 
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
from nn_model import shallow_model,deconv_model,conv_model,conv_real_model,much_complex_model,min_max_model,complex_model_DFT
from signal_utils import data_read,data_prepare,integ_fftshfit

# min_max_model is the model used to do the training 
# this file is to find the new approximation for the DFT

input_signals = loadmat('./data/data_1_1/input_signal.mat')['input_signal'][:250,:]   # read the input signal in the spatial domain
input_integs = loadmat('./data/data_1_1/input_integ.mat')['input_integ'][:250,:]  # read the integs for the input signal
output_frequencys = loadmat('./data/data_1_1/U1_DFT.mat')['U1_DFT'][:250,:]   # the real DFT result
NUDFT_frequencys = loadmat('./data/data_1_1/U1.mat')['U1'][:250,:]


input_signals = np.transpose(input_signals)
input_integs = np.transpose(input_integs).astype(np.int64)
output_frequencys = np.transpose(output_frequencys)
NUDFT_frequencys = np.transpose(NUDFT_frequencys)



print(input_signals.shape)
print(input_integs.shape)
print(output_frequencys.shape)


new_input_signals = np.zeros((5000,500)) + 1j*np.zeros((5000,500))
for j in range(5000):
    a = np.expand_dims(input_signals[j,:],0)
    b = np.expand_dims(input_integs[j,:],0)
    new_input_signals[j,:] = data_prepare(a,b,1000)



input_signals = np.expand_dims(input_signals,-1)
output_frequencys = np.expand_dims(output_frequencys,-1)
new_input_signals = np.expand_dims(new_input_signals,-1)




input_signals = np.concatenate([np.real(input_signals), np.imag(input_signals)],axis=-1)
new_input_signals = np.concatenate([np.real(new_input_signals), np.imag(new_input_signals)],axis=-1)
output_frequencys = np.concatenate([np.real(output_frequencys), np.imag(output_frequencys)],axis=-1)



lr = 1e-3
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model = min_max_model(shape=(500,2),kernel_size=9,use_bias=False)
model2 = min_max_model(shape=(500,2),kernel_size=7,use_bias=False)
model3 = min_max_model(shape=(500,2),kernel_size=5,use_bias=False)
model4 = min_max_model(shape=(500,2),kernel_size=3,use_bias=False)



model.summary()
model.load_weights('./DFT_approximation/data_1_1_model_kernel_size_9_weights.h5')
model.compile(optimzier=Adam,loss=mean_absolute_error)

model2.load_weights('./DFT_approximation/data_1_1_model_kernel_size_7_weights.h5')
model2.compile(optimzier=Adam,loss=mean_absolute_error)

model3.load_weights('./DFT_approximation/data_1_1_model_kernel_size_5_weights.h5')
model3.compile(optimzier=Adam,loss=mean_absolute_error)

model4.load_weights('./DFT_approximation/data_1_1_model_kernel_size_3_weights.h5')
model4.compile(optimzier=Adam,loss=mean_absolute_error)

output_frequencys_predictions = model.predict(new_input_signals)
output_frequencys_predictions2 = model2.predict(new_input_signals)
output_frequencys_predictions3 = model3.predict(new_input_signals)
output_frequencys_predictions4 = model4.predict(new_input_signals)




y_true = output_frequencys[:,:,0] + 1j*output_frequencys[:,:,1]
y_predicton = output_frequencys_predictions[:,:,0] + 1j*output_frequencys_predictions[:,:,1]
y_predicton2 = output_frequencys_predictions2[:,:,0] + 1j*output_frequencys_predictions2[:,:,1]
y_predicton3 = output_frequencys_predictions3[:,:,0] + 1j*output_frequencys_predictions3[:,:,1]
y_predicton4 = output_frequencys_predictions4[:,:,0] + 1j*output_frequencys_predictions4[:,:,1]


plt.figure(figsize=(8,200))
for image_number in range(50):
    plt.subplot(50,1,image_number+1)
    plt.plot(np.abs(y_true[image_number,:]),color='r',label='true result')
    plt.plot(np.abs(y_predicton[image_number,:]),color='b',label='prediction result')
    plt.plot(np.abs(y_predicton2[image_number,:]))
    plt.plot(np.abs(y_predicton3[image_number,:]))
    plt.plot(np.abs(y_predicton4[image_number,:]))
    plt.plot(np.abs(NUDFT_frequencys[image_number,:]),color='k')
    plt.legend(['true','kernel9','kernel7','kernel5','kernel3','NUFFT'],loc='best')

path = './DFT_approximation/data_1_1_MSE_complex_model_training_test1.pdf'
plt.savefig(path,format='pdf')
plt.close()

del model 
del model2
del model3
del model4
del y_predicton
del y_true
del output_frequencys_predictions
del output_frequencys_predictions2
del output_frequencys_predictions3
del output_frequencys_predictions4




# convolution in frequency domain
lr = 1e-3
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model1 = complex_model_DFT(shape=(500,2),kernel_size=3,use_bias=False)
model2 = complex_model_DFT(shape=(500,2),kernel_size=5,use_bias=False)
model3 = complex_model_DFT(shape=(500,2),kernel_size=7,use_bias=False)
model4 = complex_model_DFT(shape=(500,2),kernel_size=9,use_bias=False)



model1.summary()
model1.load_weights('./DFT_approximation/data_1_1_MAE_complex_model_DFT_kernel_size_3_weights.h5')
model1.compile(optimzier=Adam,loss=mean_absolute_error)

model2.load_weights('./DFT_approximation/data_1_1_MAE_complex_model_DFT_kernel_size_5_weights.h5')
model2.compile(optimzier=Adam,loss=mean_absolute_error)

model3.load_weights('./DFT_approximation/data_1_1_MAE_complex_model_DFT_kernel_size_7_weights.h5')
model3.compile(optimzier=Adam,loss=mean_absolute_error)

model4.load_weights('./DFT_approximation/data_1_1_MAE_complex_model_DFT_kernel_size_9_weights.h5')
model4.compile(optimzier=Adam,loss=mean_absolute_error)

output_frequencys_predictions = model1.predict(new_input_signals)
output_frequencys_predictions2 = model2.predict(new_input_signals)
output_frequencys_predictions3 = model3.predict(new_input_signals)
output_frequencys_predictions4 = model4.predict(new_input_signals)




y_true = output_frequencys[:,:,0] + 1j*output_frequencys[:,:,1]
y_predicton = output_frequencys_predictions[:,:,0] + 1j*output_frequencys_predictions[:,:,1]
y_predicton2 = output_frequencys_predictions2[:,:,0] + 1j*output_frequencys_predictions2[:,:,1]
y_predicton3 = output_frequencys_predictions3[:,:,0] + 1j*output_frequencys_predictions3[:,:,1]
y_predicton4 = output_frequencys_predictions4[:,:,0] + 1j*output_frequencys_predictions4[:,:,1]


plt.figure(figsize=(8,200))
for image_number in range(50):
    plt.subplot(50,1,image_number+1)
    plt.plot(np.abs(y_true[image_number,:]),color='r',label='true result')
    plt.plot(np.abs(y_predicton[image_number,:]),color='b',label='prediction result')
    plt.plot(np.abs(y_predicton2[image_number,:]))
    plt.plot(np.abs(y_predicton3[image_number,:]))
    plt.plot(np.abs(y_predicton4[image_number,:]))
    plt.plot(np.abs(NUDFT_frequencys[image_number,:]),color='k')
    plt.legend(['true','kernel3','kernel5','kernel7','kernel9','NUFFT'],loc='best')

path = './DFT_approximation/data_1_1_MAE_complex_model_training_test2.pdf'
plt.savefig(path,format='pdf')
plt.close()
