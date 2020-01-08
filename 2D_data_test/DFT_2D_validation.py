import numpy as np
from scipy.fftpack import fft2,ifft2,fftshift,ifftshift
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
from complex_conv import ComplexConv1D,ComplexConv2D
from nn_model import min_max_model,complex_model_DFT
from signal_utils import data_read,data_prepare,integ_fftshfit,data_prepare2D
import os




main_path = './result/1_7'
if not os.path.isdir(main_path):
    os.mkdir(main_path)



number_points = 200
Ts = 4
NU_length = int(number_points/Ts)
oversample_ratio = 2
oversample_length = NU_length*oversample_ratio

input_signals = loadmat('./1_5_data/input_signal.mat')['input_signal'][:NU_length,:NU_length,1:]
input_integs = loadmat('./1_5_data/input_integ.mat')['input_integ'][:NU_length,:NU_length,:,1:] # careful about the shape of input_integs 
output_frequencys = loadmat('./1_5_data/U1_DFT.mat')['U1_DFT'][:NU_length,:NU_length,1:]


input_signals = np.transpose(input_signals,(2,0,1))
input_integs = np.transpose(input_integs,(3,0,1,2))
output_frequencys = np.transpose(output_frequencys,(2,0,1))

print(input_signals.shape)
print(input_integs.shape)
print(output_frequencys.shape)

sample_size = len(input_signals)
#print(sample_size)
#pdb.set_trace()

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

print(new_input_signals.shape)
print(input_integs.shape)
print(output_frequencys.shape)
del input_signals



model1 = min_max_model(shape=(oversample_length,oversample_length,2),kernel_size=3)
model1.compile(optimzier=Adam,loss=mean_squared_error)
model1.load_weights('./model/1_6/data_1_5_model_kernel_size_3_weights.h5')


model2 = min_max_model(shape=(oversample_length,oversample_length,2),kernel_size=5)
model2.compile(optimzier=Adam,loss=mean_squared_error)
model2.load_weights('./model/1_6/data_1_5_model_kernel_size_5_weights.h5')



model3 = min_max_model(shape=(oversample_length,oversample_length,2),kernel_size=7)
model3.compile(optimzier=Adam,loss=mean_squared_error)
model3.load_weights('./model/1_6/data_1_5_model_kernel_size_7_weights.h5')


model4 = min_max_model(shape=(oversample_length,oversample_length,2),kernel_size=9)
model4.compile(optimzier=Adam,loss=mean_squared_error)
model4.load_weights('./model/1_6/data_1_5_model_kernel_size_9_weights.h5')



output_frequencys_predictions = model1.predict(new_input_signals)
output_frequencys_predictions2 = model2.predict(new_input_signals)
output_frequencys_predictions3 = model3.predict(new_input_signals)
output_frequencys_predictions4 = model4.predict(new_input_signals)


y_true = output_frequencys[:,:,:,0] + 1j*output_frequencys[:,:,:,1]
y_prediction = output_frequencys_predictions[:,:,:,0] + 1j*output_frequencys_predictions[:,:,:,1]
y_prediction2 = output_frequencys_predictions2[:,:,:,0] + 1j*output_frequencys_predictions2[:,:,:,1]
y_prediction3 = output_frequencys_predictions3[:,:,:,0] + 1j*output_frequencys_predictions3[:,:,:,1]
y_prediction4 = output_frequencys_predictions4[:,:,:,0] + 1j*output_frequencys_predictions4[:,:,:,1]




result_dir1 = './result/1_6/kernel_3/'
if not os.path.isdir(result_dir1):
    os.mkdir(result_dir1)


result_dir2 = './result/1_6/kernel_5'
if not os.path.isdir(result_dir2):
    os.mkdir(result_dir2)

result_dir3 = './result/1_6/kernel_7'
if not os.path.isdir(result_dir3):
    os.mkdir(result_dir3)


result_dir4 = './result/1_6/kernel_9'
if not os.path.isdir(result_dir4):
    os.mkdir(result_dir4)


# kernel size is 3
for image_number in range(30):
    plt.figure(figsize=(4,200))
    fig,axs = plt.subplots(nrows=2,ncols=2)
    axs[0,0].set_title('True DFT result')
    axs[0,1].set_title('Prediction DFT result')
    real_true=axs[0,0].imshow(np.real(y_true[image_number,:,:]),cmap='bwr', label='True')
    fig.colorbar(real_true, ax=axs[0,0])
    
    real_pred=axs[0,1].imshow(np.real(y_prediction[image_number,:,:]),cmap='bwr', label='Pred')
    fig.colorbar(real_pred, ax=axs[0,1])
    
    imag_true=axs[1,0].imshow(np.imag(y_true[image_number,:,:]),cmap='bwr', label='True')
    fig.colorbar(imag_true, ax=axs[1,0])
    
    imag_pred=axs[1,1].imshow(np.imag(y_prediction[image_number,:,:]),cmap='bwr', label='Pred')
    fig.colorbar(imag_pred, ax=axs[1,1])
    path =result_dir1+'/'+str(image_number+1)+'.jpg'
    fig.savefig(path,format='jpg')
    plt.close('all')

# kernel size is 5
for image_number in range(30):
    plt.figure(figsize=(4,200))
    fig,axs = plt.subplots(nrows=2,ncols=2)
    axs[0,0].set_title('True DFT result')
    axs[0,1].set_title('Prediction DFT result')
    real_true=axs[0,0].imshow(np.real(y_true[image_number,:,:]),cmap='bwr', label='True')
    fig.colorbar(real_true, ax=axs[0,0])
    
    real_pred=axs[0,1].imshow(np.real(y_prediction2[image_number,:,:]),cmap='bwr', label='Pred')
    fig.colorbar(real_pred, ax=axs[0,1])
    
    imag_true=axs[1,0].imshow(np.imag(y_true[image_number,:,:]),cmap='bwr', label='True')
    fig.colorbar(imag_true, ax=axs[1,0])
    
    imag_pred=axs[1,1].imshow(np.imag(y_prediction2[image_number,:,:]),cmap='bwr', label='Pred')
    fig.colorbar(imag_pred, ax=axs[1,1])
    path =result_dir2+'/'+str(image_number+1)+'.jpg'
    fig.savefig(path,format='jpg')
    plt.close('all')

# kernel size is 7
for image_number in range(30):
    plt.figure(figsize=(4,200))
    fig,axs = plt.subplots(nrows=2,ncols=2)
    axs[0,0].set_title('True DFT result')
    axs[0,1].set_title('Prediction DFT result')
    real_true=axs[0,0].imshow(np.real(y_true[image_number,:,:]),cmap='bwr', label='True')
    fig.colorbar(real_true, ax=axs[0,0])
    
    real_pred=axs[0,1].imshow(np.real(y_prediction3[image_number,:,:]),cmap='bwr', label='Pred')
    fig.colorbar(real_pred, ax=axs[0,1])
    
    imag_true=axs[1,0].imshow(np.imag(y_true[image_number,:,:]),cmap='bwr', label='True')
    fig.colorbar(imag_true, ax=axs[1,0])
    
    imag_pred=axs[1,1].imshow(np.imag(y_prediction3[image_number,:,:]),cmap='bwr', label='Pred')
    fig.colorbar(imag_pred, ax=axs[1,1])
    path =result_dir3+'/'+str(image_number+1)+'.jpg'
    fig.savefig(path,format='jpg')
    plt.close('all')

# kernel size is 9
for image_number in range(30):
    plt.figure(figsize=(4,200))
    fig,axs = plt.subplots(nrows=2,ncols=2)
    axs[0,0].set_title('True DFT result')
    axs[0,1].set_title('Prediction DFT result')
    real_true=axs[0,0].imshow(np.real(y_true[image_number,:,:]),cmap='bwr', label='True')
    fig.colorbar(real_true, ax=axs[0,0])
    
    real_pred=axs[0,1].imshow(np.real(y_prediction4[image_number,:,:]),cmap='bwr', label='Pred')
    fig.colorbar(real_pred, ax=axs[0,1])
    
    imag_true=axs[1,0].imshow(np.imag(y_true[image_number,:,:]),cmap='bwr', label='True')
    fig.colorbar(imag_true, ax=axs[1,0])
    
    imag_pred=axs[1,1].imshow(np.imag(y_prediction4[image_number,:,:]),cmap='bwr', label='Pred')
    fig.colorbar(imag_pred, ax=axs[1,1])
    path =result_dir4+'/'+str(image_number+1)+'.jpg'
    fig.savefig(path,format='jpg')
    plt.close('all')


