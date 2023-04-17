"""
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
"""

from tensorflow.python.keras import layers
from tensorflow.python.keras import models

from training.conv_blocks import conv_block



def get_cnn_model(inputs):
    """.
    

    Parameters
    ----------
    inputs : TYPE
        DESCRIPTION.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """
    normalized_input = layers.BatchNormalization()(inputs)  # 257x257x?
    conv1 = conv_block(normalized_input, 32)  # 128x128x32
    conv2 = conv_block(conv1, 64)  # 64x64x64
    conv3 = conv_block(conv2, 128)  # 32x32x128
    conv4 = conv_block(conv3, 256)  # 16x16x256
    conv5 = conv_block(conv4, 256)  # 8x8x256
    conv6 = conv_block(conv5, 64)  # 4x4x64
    conv7 = conv_block(conv6, 32)  # 2x2x32
    output = conv_block(conv7, 1)  # 1x1x1
    model = models.Model(inputs=[inputs], outputs=[output])

    return model
