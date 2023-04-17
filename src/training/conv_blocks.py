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


def conv_block(input_tensor, num_filters, kernel_size=(3, 3), normalize=True,
               activation='tanh'):
    conv1 = layers.Conv2D(num_filters, kernel_size,
                          padding='same')(input_tensor)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation(activation)(conv1)
    conv2 = layers.Conv2D(num_filters, kernel_size, padding='same')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation(activation)(conv2)
    pooled = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2)
    return pooled

def spectral_block(spectral_input):
    normalized_input = layers.BatchNormalization()(spectral_input)  # 257x257x5
    conv1 = conv_block(normalized_input, 32) # 128x128x32
    conv2 = encoder_pool(conv1, 64) # 64x64x64
    conv3 = encoder_pool(conv2, 128) # 32x32x128
    encoder3 = encoder_pool(encoder2, 256) # 16x16x256
    encoder4 = encoder_pool(encoder3, 256) # 8x8x256
    encoder5 = encoder_pool(encoder4, 128) # 4x4x128
    encoder6 = encoder_pool(encoder5, 64) # 2x2x64
    encoder7 = encoder_pool(encoder6, 32) # 1x1x32
    conv1 = conv_block(encoder7, 16, (1,1)) # 1x1x16
    conv2 = conv_block(conv1, 1,(1,1)) # 1x1x1