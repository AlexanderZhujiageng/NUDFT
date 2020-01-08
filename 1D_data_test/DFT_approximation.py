# In[]
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


# In[]


input_signals = np.expand_dims(input_signals,-1)
output_frequencys = np.expand_dims(output_frequencys,-1)
new_input_signals = np.expand_dims(new_input_signals,-1)




input_signals = np.concatenate([np.real(input_signals), np.imag(input_signals)],axis=-1)
new_input_signals = np.concatenate([np.real(new_input_signals), np.imag(new_input_signals)],axis=-1)
output_frequencys = np.concatenate([np.real(output_frequencys), np.imag(output_frequencys)],axis=-1)

'''

lr = 1e-3
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model = min_max_model(shape=(500,2),kernel_size=9,use_bias=False)
model.summary()
model.compile(optimzier=Adam,loss=mean_squared_error)
model.fit(new_input_signals,output_frequencys,batch_size=32,epochs=10000,verbose=1,callbacks=[lr_adapt,early_stop])
model.save_weights('./DFT_approximation/data_1_1_model_kernel_size_9_weights.h5')

# Model 2 the kernel size is 7
lr = 1e-3
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model2 = min_max_model(shape=(500,2),kernel_size=7,use_bias=False)
model2.summary()
model2.compile(optimzier=Adam,loss=mean_squared_error)
model2.fit(new_input_signals,output_frequencys,batch_size=32,epochs=10000,verbose=1,callbacks=[lr_adapt,early_stop])
model2.save_weights('./DFT_approximation/data_1_1_model_kernel_size_7_weights.h5')

# Model 3 the kernel size is 5
lr = 1e-3
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model3 = min_max_model(shape=(500,2),kernel_size=5,use_bias=False)
model3.summary()
model3.compile(optimzier=Adam(lr=lr),loss=mean_squared_error)
model3.fit(new_input_signals,output_frequencys,batch_size=32,epochs=10000,verbose=1,callbacks=[lr_adapt,early_stop])
model3.save_weights('./DFT_approximation/data_1_1_model_kernel_size_5_weights.h5')

# Model 4 the kernel size is 3
lr = 1e-3
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model4 = min_max_model(shape=(500,2),kernel_size=3,use_bias=False)
model4.summary()
model4.compile(optimzier=Adam(lr=lr),loss=mean_squared_error)
model4.fit(new_input_signals,output_frequencys,batch_size=32,epochs=10000,verbose=1,callbacks=[lr_adapt,early_stop])
model4.save_weights('./DFT_approximation/data_1_1_model_kernel_size_3_weights.h5')



del model
del model2
del model3
del model4
'''
######################################################################
################ Alternative training below###########################
###################################################################### 


# kernel size is 3
model1 = min_max_model(shape=(500,2),kernel_size=3,use_bias=False)
model1.summary()
lr = 1e-3
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model1.compile(optimzier=Adam(lr=lr),loss=mean_squared_error)
for iteration_times in range(50):
    number_layer = 0
    for layer in model1.layers:
        if np.mod(iteration_times,2)==0:
            # even and begining time, set the first scalar layer weight to be fixed
            if number_layer==1:
                layer.trainable = False 
            if number_layer==2:
                layer.trainable = True    
        else:
            # odd time, set the convolution layer weight to be fixed
            if number_layer==1:
                layer.trainable = True
            if number_layer==2:
                layer.trainable = False
        number_layer +=1
    model1.compile(optimizer=Adam(lr=lr),loss=mean_squared_error)
    model1.fit(new_input_signals,output_frequencys,batch_size=32,epochs=1000,callbacks=[lr_adapt,early_stop])

model1_alterative_path = './DFT_approximation/data_1_1_model_kernel_size_3_mse_alterative'
model1.save_weights(model1_alterative_path+'_weights.h5')



# kernel size is 5
model2 = min_max_model(shape=(500,2),kernel_size=5,use_bias=False)
model2.summary()
lr = 1e-3
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model2.compile(optimzier=Adam(lr=lr),loss=mean_squared_error)
for iteration_times in range(50):
    number_layer = 0
    for layer in model2.layers:
        if np.mod(iteration_times,2)==0:
            # even and begining time, set the first scalar layer weight to be fixed
            if number_layer==1:
                layer.trainable = False 
            if number_layer==2:
                layer.trainable = True    
        else:
            # odd time, set the convolution layer weight to be fixed
            if number_layer==1:
                layer.trainable = True
            if number_layer==2:
                layer.trainable = False
        number_layer +=1
    model2.compile(optimizer=Adam(lr=lr),loss=mean_squared_error)
    model2.fit(new_input_signals,output_frequencys,batch_size=32,epochs=1000,callbacks=[lr_adapt,early_stop])

model2_alterative_path = './DFT_approximation/data_1_1_model_kernel_size_5_mse_alterative'
model2.save_weights(model2_alterative_path+'_weights.h5')

# kernel size is 7
model3 = min_max_model(shape=(500,2),kernel_size=7,use_bias=False)
model3.summary()
lr = 1e-3
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model3.compile(optimzier=Adam(lr=lr),loss=mean_squared_error)
for iteration_times in range(50):
    number_layer = 0
    for layer in model3.layers:
        if np.mod(iteration_times,2)==0:
            # even and begining time, set the first scalar layer weight to be fixed
            if number_layer==1:
                layer.trainable = False 
            if number_layer==2:
                layer.trainable = True    
        else:
            # odd time, set the convolution layer weight to be fixed
            if number_layer==1:
                layer.trainable = True
            if number_layer==2:
                layer.trainable = False
        number_layer +=1
    model3.compile(optimizer=Adam(lr=lr),loss=mean_squared_error)
    model3.fit(new_input_signals,output_frequencys,batch_size=32,epochs=1000,callbacks=[lr_adapt,early_stop])

model3_alterative_path = './DFT_approximation/data_1_1_model_kernel_size_7_mse_alterative'
model3.save_weights(model3_alterative_path+'_weights.h5')

# kernel size is 9
model4 = min_max_model(shape=(500,2),kernel_size=9,use_bias=False)
model4.summary()
lr = 1e-3
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model4.compile(optimzier=Adam(lr=lr),loss=mean_squared_error)
for iteration_times in range(50):
    number_layer = 0
    for layer in model4.layers:
        if np.mod(iteration_times,2)==0:
            # even and begining time, set the first scalar layer weight to be fixed
            if number_layer==1:
                layer.trainable = False 
            if number_layer==2:
                layer.trainable = True    
        else:
            # odd time, set the convolution layer weight to be fixed
            if number_layer==1:
                layer.trainable = True
            if number_layer==2:
                layer.trainable = False
        number_layer +=1
    model4.compile(optimizer=Adam(lr=lr),loss=mean_squared_error)
    model4.fit(new_input_signals,output_frequencys,batch_size=32,epochs=1000,callbacks=[lr_adapt,early_stop])

model4_alterative_path = './DFT_approximation/data_1_1_model_kernel_size_9_mse_alterative'
model4.save_weights(model3_alterative_path+'_weights.h5')

'''
# convolution in frequency kernel size 3
lr = 1e-3
model1 = complex_model_DFT(kernel_size=3)
model1.summary()
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model1.compile(optimzier=Adam(lr=lr),loss=mean_absolute_error)
model1.fit(new_input_signals,output_frequencys,batch_size=32,epochs=10000,verbose=1,callbacks=[lr_adapt,early_stop])
model1.save_weights('./DFT_approximation/data_1_1_MAE_complex_model_DFT_kernel_size_3_weights.h5')


# convolution in frequency kernel size 5
lr = 1e-3
model2 = complex_model_DFT(kernel_size=5)
model2.summary()
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model2.compile(optimzier=Adam(lr=lr),loss=mean_absolute_error)
model2.fit(new_input_signals,output_frequencys,batch_size=32,epochs=10000,verbose=1,callbacks=[lr_adapt,early_stop])
model2.save_weights('./DFT_approximation/data_1_1_MAE_complex_model_DFT_kernel_size_5_weights.h5')


# convolution in frequency kernel size 7
lr = 1e-3
model3 = complex_model_DFT(kernel_size=7)
model3.summary()
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model3.compile(optimzier=Adam(lr=lr),loss=mean_absolute_error)
model3.fit(new_input_signals,output_frequencys,batch_size=32,epochs=10000,verbose=1,callbacks=[lr_adapt,early_stop])
model3.save_weights('./DFT_approximation/data_1_1_MAE_complex_model_DFT_kernel_size_7_weights.h5')


# convolution in frequency kernel size 9
lr = 1e-3
model4 = complex_model_DFT(kernel_size=9)
model4.summary()
lr_adapt = ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,verbose=1,min_lr=1e-7)
early_stop = EarlyStopping(monitor='loss',min_delta=1e-4,patience=20,verbose=1,restore_best_weights=True)
model4.compile(optimzier=Adam(lr=lr),loss=mean_absolute_error)
model4.fit(new_input_signals,output_frequencys,batch_size=32,epochs=10000,verbose=1,callbacks=[lr_adapt,early_stop])
model4.save_weights('./DFT_approximation/data_1_1_MAE_complex_model_DFT_kernel_size_9_weights.h5')
'''