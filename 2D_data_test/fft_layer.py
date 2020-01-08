import numpy as np 
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Lambda, Layer, InputSpec
from tensorflow.keras.models import Model 
from tensorflow.keras import activations, initializers, regularizers,constraints
import tensorflow as tf 

class fft_layer1D(Layer):
    def __init__(self,seed=None,**kwargs):
        super(fft_layer1D,self).__init__(**kwargs)
        self.seed = seed
    
    def build(self,input_shape):
        pass

    def call(self,inputs):
        real_inputs = inputs[:,:,0]
        imag_inputs = inputs[:,:,1]
        inputs_complex = tf.complex(real_inputs,imag_inputs)
        outputs_complex = tf.signal.fft(inputs_complex)
        outputs_complex = tf.signal.fftshift(outputs_complex,-1)
        outputs_complex_middle = outputs_complex[:,25:75]
        outputs_complex_middle = tf.signal.ifftshift(outputs_complex_middle,-1)
        output_real = tf.expand_dims(tf.math.real(outputs_complex_middle),-1)
        output_imag = tf.expand_dims(tf.math.imag(outputs_complex_middle),-1)
        output = K.concatenate([output_real,output_imag],axis=-1)
        return output

    def compute_output_shape(self,input_shape):
        return input_shape

class fft_layer2D(Layer):
    def __init__(self,seed=None,**kwargs):
        super(fft_layer2D,self).__init__(**kwargs)
        self.seed=seed
    
    def build(self,input_shape):
        pass

    def call(self,inputs):
        real_inputs = inputs[:,:,:,0]
        imag_inputs = inputs[:,:,:,1]
        inputs_complex = tf.complex(real_inputs,imag_inputs)
        outputs_complex = tf.signal.fft2d(inputs_complex)
        outputs_complex = tf.signal.fftshift(outputs_complex,(-2,-1))
        outputs_complex_middle = outputs_complex[:,25:75,25:75]
        outputs_complex_middle = tf.signal.ifftshift(outputs_complex_middle,(-2,-1))
        output_real = tf.expand_dims(tf.math.real(outputs_complex_middle),-1)
        output_imag = tf.expand_dims(tf.math.imag(outputs_complex_middle),-1)
        output = K.concatenate([output_real,output_imag],axis=-1)
        return output
