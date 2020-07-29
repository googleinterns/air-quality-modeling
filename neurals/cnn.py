from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import metrics
from tensorflow.python.keras import optimizers


def conv_block(input_tensor, num_filters, kernel_size=(3,3)):
	encoder = layers.Conv2D(num_filters, kernel_size, padding='same')(input_tensor)
	encoder = layers.BatchNormalization()(encoder)
	encoder = layers.Activation('relu')(encoder)
	encoder = layers.Conv2D(num_filters, kernel_size, padding='same')(encoder)
	encoder = layers.BatchNormalization()(encoder)
	encoder = layers.Activation('relu')(encoder)
	return encoder

def encoder_pool(input_tensor, num_filters):
	encoder = conv_block(input_tensor, num_filters)
	encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
	return encoder_pool

def get_model(input_shape, optimizer_name, loss_name, metric_names):
    inputs = layers.Input(shape=input_shape) # 257x257x?
    normalized_input = layers.BatchNormalization()(inputs)
    encoder0 = encoder_pool(normalized_input, 32) # 128x128x32
    encoder1 = encoder_pool(encoder0, 64) # 64x64x64
    encoder2 = encoder_pool(encoder1, 128) # 32x32x128
    encoder3 = encoder_pool(encoder2, 256) # 16x16x256
    encoder4 = encoder_pool(encoder3, 256) # 8x8x256
    encoder5 = encoder_pool(encoder4, 128) # 4x4x128
    encoder6 = encoder_pool(encoder5, 64) # 2x2x64
    encoder7 = encoder_pool(encoder6, 32) # 1x1x32
    conv1 = conv_block(encoder7, 16, (1,1)) # 1x1x16
    conv2 = conv_block(conv1, 1,(1,1)) # 1x1x1
    outputs = conv2
    model = models.Model(inputs=[inputs], outputs=[outputs])
    
    model.compile(
		optimizer=optimizers.get(optimizer_name), 
		loss=losses.get(loss_name),
		metrics=[metrics.get(metric) for metric in metric_names])
    
    return model