
# In[]
import numpy as np
import tensorflow as tf 
import tensorflow.keras as keras 
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Conv1D,Input,Dropout,concatenate,Conv2D
from complex_deconv_freq import Complex_deconv
from complex_conv import ComplexConv1D,ComplexConv2D
from fft_layer import fft_layer1D,fft_layer2D
from tensorflow.keras.initializers import TruncatedNormal, glorot_normal,Ones
#from interpolation_layer import interpolation_layer1D

# In[]


def min_max_model(shape=(500,500,2),kernel_size=3,use_bias=False):
    input_layer = Input(shape=shape)
    scalar_layer = Complex_deconv(use_bias=use_bias,real_kernel_initializer='ones')(input_layer)
    conv1 = ComplexConv2D(filters=1,kernel_size=kernel_size,padding='same',use_bias=use_bias)(scalar_layer)
    fft1 = fft_layer2D()(conv1)
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



model = min_max_model()
model.summary()
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
