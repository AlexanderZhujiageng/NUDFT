import numpy as np 
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Lambda, Layer, InputSpec
import conv_utils
from tensorflow.keras.models import Model 
from tensorflow.keras import activations, initializers, regularizers,constraints
import tensorflow as tf 


class Complex_deconv(Layer):
    def __init__(self,
                activation=None,
                use_bias = False,
                real_kernel_initializer='uniform',
                imag_kernel_initializer='zeros',
                kernel_regularizer=None,
                bias_initializer='zeros',
                seed=None,
                **kwargs):
        super(Complex_deconv,self).__init__(**kwargs)
        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.real_kernel_initializer = initializers.get(real_kernel_initializer)
        self.imag_kernel_initializer = initializers.get(imag_kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        if seed == None:
            self.seed = np.random.randint(1,10e6)
        else:
            self.seed = seed 
    
    def build(self,input_shape):
        input_dim = input_shape[-1]//2
        shape = input_shape[1:-1] + (input_dim,)
        self.shape = shape 
        self.kernel_shape = shape
        self.real_kernel = self.add_weight(shape=self.kernel_shape,
                                        initializer=self.real_kernel_initializer,
                                        name='real_kernel',
                                        regularizer=self.kernel_regularizer)
        self.imag_kernel = self.add_weight(shape=self.kernel_shape,
                                        initializer=self.imag_kernel_initializer,
                                        name='imag_kernel',
                                        regularizer=self.kernel_regularizer)
        if self.use_bias:
            bias_shape = input_shape[1:-1] + (input_dim*2,) 
            self.bias = self.add_weight(shape=bias_shape,initializer=self.bias_initializer,name='bias')
        else:
            self.bias = None 
        self.build = True 

    def call(self,inputs):
        shape = K.shape(inputs)
        inputs_real = tf.expand_dims(inputs[:,:,0],axis=-1)
        inputs_imag = tf.expand_dims(inputs[:,:,1],axis=-1)

        outputs_real = tf.math.multiply(inputs_real,self.real_kernel) - tf.math.multiply(inputs_imag,self.imag_kernel)
        outputs_imag = tf.math.multiply(inputs_real,self.imag_kernel) + tf.math.multiply(inputs_imag,self.real_kernel)


        outputs = K.concatenate([outputs_real,outputs_imag],axis=-1)
        if self.use_bias:
            outputs = K.bias_add(outputs,self.bias,data_format='channels_last')
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs
    
    def compute_output_shape(self,input_shape):
        return input_shape

        

