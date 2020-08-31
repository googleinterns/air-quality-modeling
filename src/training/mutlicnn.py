'''
Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''


from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import metrics
from tensorflow.python.keras import optimizers
#from tensorflow.python.keras import activations

def conv_block(input_tensor, num_filters, kernel_size=(3,3),
                               normalize = True,                    
                               activation='softplus'):
	encoder = layers.Conv2D(num_filters, kernel_size, padding='same')(input_tensor)
	encoder = layers.BatchNormalization()(encoder)
	encoder = layers.Activation(activation)(encoder)
	encoder = layers.Conv2D(num_filters, kernel_size, padding='same')(encoder)
	encoder = layers.BatchNormalization()(encoder)
	encoder = layers.Activation(activation)(encoder)
	return encoder

def encoder_pool(input_tensor, num_filters):
	encoder = conv_block(input_tensor, num_filters)
	encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
	return encoder_pool

def reduction_block(inp):
    normalized_input = layers.BatchNormalization()(inp)
    encoder0 = encoder_pool(normalized_input, 32) # 128x128x32
    encoder1 = encoder_pool(encoder0, 64) # 64x64x64
    encoder2 = encoder_pool(encoder1, 128) # 32x32x128
    encoder3 = encoder_pool(encoder2, 256) # 16x16x256
    encoder4 = encoder_pool(encoder3, 128) # 8x8x256
    encoder5 = encoder_pool(encoder4, 16) # 4x4x16
    return encoder5

def get_model(inputs):
    outputs = [reduction_block(inp) for inp in inputs]
    encoder = layers.Concatenate(axis=-1)(outputs)
    
    encoder0 = encoder_pool(encoder, 32) # 4x4x32
    encoder1 = encoder_pool(encoder0, 64) # 2x2x64
    encoder2 = encoder_pool(encoder1, 1) # 1x1x1

    model = models.Model(inputs=[inputs], outputs=[encoder2])
    
    return model