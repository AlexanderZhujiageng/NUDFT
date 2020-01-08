
# In[]
import numpy as np
import tensorflow as tf 
import tensorflow.keras as keras 
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Conv1D,Input,Dropout,concatenate
from complex_deconv_freq import Complex_deconv
from complex_conv import ComplexConv1D
from fft_layer import fft_layer1D
from complex_conv_real_kernel import Complex_real_Conv1D
from tensorflow.keras.initializers import TruncatedNormal, glorot_normal,Ones
#from interpolation_layer import interpolation_layer1D

# In[]
def shallow_model(shape=(250,2),use_bias=False):
    input_layer = Input(shape=(250,2))
    conv1 = ComplexConv1D(filters=1,
                          kernel_size=30,
                          padding='same',kernel_initializer='glorot_normal',use_bias=True,activation='linear')(input_layer)
    #conv2 = ComplexConv1D(filters=32,kernel_size=3,padding='same', kernel_initializer='glorot_normal',use_bias=True,activation='linear')(conv1)
    #conv3 = ComplexConv1D(filters=16,kernel_size=3,padding='same',kernel_initializer='glorot_normal',use_bias=False,activation='linear')(conv2)
    #conv4 = ComplexConv1D(1,1,padding='same',activation='linear')(conv3)
    model = Model(input_layer,conv1)
    return model

def deconv_model(shape=(250,2),use_bias=False):
    input_layer = Input(shape=shape)
    deconv1 = Complex_deconv(use_bias=use_bias)(input_layer)
    model = Model(input_layer,deconv1)
    return model


def conv_model(shape=(500,2),use_bias=True):
    '''
    to make sure that the input here is 
    '''
    input_layer = Input(shape=shape)
    conv1 = ComplexConv1D(filters=100,kernel_size=5,padding='same',activation='linear',kernel_initializer=TruncatedNormal(mean=0.33))(input_layer)
    conv2 = ComplexConv1D(filters=50,kernel_size=3,padding='same',activation='linear',kernel_initializer=TruncatedNormal(mean=0.33))(conv1)
    conv3 = ComplexConv1D(filters=1,kernel_size=1,padding='same',activation='linear',use_bias=False)(conv2)
    fft1 = fft_layer1D()(conv3)
    model = Model(input_layer,fft1)
    return model 

def conv_real_model(shape=(500,2),use_bias=False):
    '''
    This model is similar to the conv_model, but the kernel itself is real number
    '''
    input_layer = Input(shape=shape)
    conv1 = Complex_deconv(use_bias=use_bias,real_kernel_initializer='ones')(input_layer)
    #conv1 = Complex_real_Conv1D(filters=1,kernel_size=3,padding='same',activation='linear', kernel_initializer=TruncatedNormal(mean=0.33),use_bias=False)(input_layer)
    #conv2 = Complex_real_Conv1D(filters=50,kernel_size=3,padding='same',activation='linear', kernel_initializer=TruncatedNormal(mean=0.33),use_bias=False)(conv1)
    #conv3 = Complex_real_Conv1D(filters=1,kernel_size=1,padding='same',activation='linear',use_bias=False)(conv2)
    fft1 = fft_layer1D()(conv1)
    model = Model(input_layer,fft1)
    return model 


def much_complex_model(shape=(500,2),use_bias=False):
    input_layer = Input(shape=shape)
    conv1 = ComplexConv1D(filters=64,kernel_size=3,padding='same',use_bias=use_bias)(input_layer)
    conv2 = ComplexConv1D(filters=128,kernel_size=3,padding='same')(conv1)
    drop1 = Dropout(0.3)(conv2)
    conv3 = ComplexConv1D(filters=256,kernel_size=3,padding='same')(drop1)
    drop2 = Dropout(0.4)(conv3)

    conv4 = ComplexConv1D(filters=128,kernel_size=3,padding='same')(drop2)
    merge1 = concatenate([conv2,conv4],axis=2)

    conv5 = ComplexConv1D(filters=64,kernel_size=3,padding='same')(merge1)
    merge2 = concatenate([conv1,conv5])


    fft1 = fft_layer1D()(merge2)
    output = ComplexConv1D(1,1,padding='same')(fft1)
    model = Model(input_layer,output)
    return model 



def min_max_model(shape=(500,2),kernel_size=9,use_bias=False):
    input_layer = Input(shape=shape)
    scalar_layer = Complex_deconv(use_bias=use_bias,real_kernel_initializer='ones')(input_layer)
    conv1 = ComplexConv1D(filters=1,kernel_size=kernel_size,padding='same',use_bias=use_bias)(scalar_layer)
    fft1 = fft_layer1D()(conv1)
    deconv1 = Complex_deconv(use_bias=use_bias,real_kernel_initializer='ones')(fft1)
    model = Model(input_layer,deconv1)
    return model 

def complex_model_DFT(shape=(500,2),kernel_size=5,use_bias=False):
    input_layer = Input(shape=shape)
    scalar_layer = Complex_deconv(use_bias=use_bias,real_kernel_initializer='ones')(input_layer)
    conv1 = ComplexConv1D(filters=1,kernel_size=kernel_size,padding='same',use_bias=use_bias)(scalar_layer)
    fft1 = fft_layer1D()(conv1)
    deconv1 = Complex_deconv(use_bias=use_bias,real_kernel_initializer='ones')(fft1)
    deconv2 = ComplexConv1D(filters=1,kernel_size=kernel_size,padding='same',use_bias=use_bias)(deconv1)
    model = Model(input_layer,deconv2)
    return model 

'''
def interpolation_model(shape=(250,3)):
    input_layer = Input(shape=shape)
    inter_layer1 = interpolation_layer1D()(input_layer)
    model = Model(input_layer,inter_layer1)
    return model 

model = interpolation_model()
model.summary()
'''

'''
model = min_max_model()
model.summary()

number_layer = 0
for layer in model.layers:
    if number_layer == 1:
        layer.trainable=False
    print(layer.name)
    print(layer.trainable)
    number_layer +=1
    
print(number_layer)

model = much_complex_model()
model.summary()
    
model1 = shallow_model()
model1.summary()


model2 = deconv_model()
model2.summary()

model3 = conv_model()
model3.summary()


model4 = conv_real_model()
model4.summary()
'''



# %%
