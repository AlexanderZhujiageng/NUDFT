# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:41:18 2018

@author: Jiageng Xhu
@usage: Complex conv
"""
from tensorflow.keras import backend as K
from tensorflow.keras import activations,initializers,regularizers,constraints
from tensorflow.keras.layers import Lambda, Layer,InputSpec,Convolution1D,Convolution2D,add,multiply,Activation,Input,concatenate
import conv_utils
from tensorflow.keras.models import Model
import numpy as np
import itertools





class Complex_real_Conv(Layer):
    def __init__(self,rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='uniform',
                 kernel_regularizer = None,
                 bias_initializer='zeros',
                 seed=None,
                 **kwargs):
        super(Complex_real_Conv,self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size,rank,'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides,rank,'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate,rank,'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        if seed is None:
            self.seed = np.random.randint(1,10e6)
        else:
            self.seed = seed
        self.input_spec = InputSpec(ndim=self.rank + 2)
    
    def build(self,input_shape):
        input_dim = input_shape[-1] // 2
        self.kernel_shape = self.kernel_size+(input_dim,self.filters)
        self.real_kernel = self.add_weight(
                shape=self.kernel_shape,
                initializer=self.kernel_initializer,
                name='real_kernel',
                regularizer=self.kernel_regularizer)
        self.imag_kernel = self.add_weight(
                shape=self.kernel_shape,
                initializer = initializers.get('zero'),
                name='imag_kernel',
                regularizer=self.kernel_regularizer,
                trainable=False)
        if self.use_bias:
            bias_shape = (2*self.filters,)
            self.bias = self.add_weight(
                    shape=bias_shape,
                    initializer=self.bias_initializer,
                    name='bias')
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=self.rank+2,
                                    axes={-1:input_dim * 2})
        self.build = True
        
    def call(self,inputs):
        input_dim = K.shape(inputs)[-1] // 2
        if self.rank == 1:
            f_real = self.real_kernel
            f_imag = self.imag_kernel
        if self.rank == 2:
            f_real = self.real_kernel
            f_imag = self.imag_kernel
        if self.rank == 3:
            f_real = self.real_kernel
            f_imag = self.imag_kernel
         
        convArgs = {"strides": self.strides[0] if self.rank == 1 else self.strides,
                    "padding": self.padding,
                    "data_format": "channels_last",
                    "dilation_rate": self.dilation_rate[0] if self.rank == 1 else self.dilation_rate
                }
        convFunc = {1: K.conv1d,
                    2: K.conv2d,
                    3: K.conv3d}[self.rank]
        #Performing the complex convolution
        
        f_real._keras_shape = self.kernel_shape
        f_imag._keras_shape = self.kernel_shape
        
        cat_kernels_4_real = K.concatenate([f_real,-f_imag],axis=-2)
        cat_kernels_4_imag = K.concatenate([f_imag, f_real],axis=-2)
        cat_kernels_4_complex = K.concatenate([cat_kernels_4_real,cat_kernels_4_imag],axis=-1)
        cat_kernels_4_complex._keras_shape = self.kernel_size + (2*input_dim,2*self.filters)
        
        output = convFunc(inputs,cat_kernels_4_complex,**convArgs)
        
        if self.use_bias:
            output = K.bias_add(
                    output,
                    self.bias,
                    data_format = "channels_last")
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i]
            )
            new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (2 * self.filters,)

class Complex_real_Conv1D(Complex_real_Conv):
    def __init__(self,filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='uniform',
                 kernel_regularizer=None,
                 bias_initializer = 'zeros',
                 seed = None,
                 **kwargs):
        super(Complex_real_Conv1D,self).__init__(
                rank = 1,
                filters=filters,
                kernel_size = kernel_size,
                strides = strides,
                padding = padding,
                dilation_rate=dilation_rate,
                activation = activation,
                use_bias = use_bias,
                kernel_initializer = kernel_initializer,
                kernel_regularizer = kernel_regularizer,
                bias_initializer = bias_initializer,
                **kwargs)