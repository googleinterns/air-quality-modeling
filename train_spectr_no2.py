import tensorflow as tf

INPUT = ['patch'+s for s in 'BGRNP']
OUTPUT = ['NO2_column_number_density']
FEATURES = INPUT+OUTPUT
KERNEL_SIZE = 128*2+1
KERNEL_SHAPE=[KERNEL_SIZE,KERNEL_SIZE]
COLUMNS = [
  tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in INPUT
] + [
  tf.io.FixedLenFeature(shape=[1,1], dtype=tf.float32) for k in OUTPUT
] 
FEATURES_DICT = dict(zip(FEATURES, COLUMNS))

FOLDER = ""

EPOCHS = 50
BTACH_SIZE  = 32

#folder = "/usr/local/google/home/aghriss/airqual/datasets"
#dataset = load_tfrecords(folder, FEATURES_DICT, INPUT, OUTPUT)

OPTIMIZER = 'SGD'
LOSS = 'MeanSquaredError'
METRICS = ['RootMeanSquaredError']

import utils
from neurals import cnn

training = utils.load_tfrecords(FOLDER, FEATURES_DICT, INPUT, OUTPUT)
model = cnn.get_model(KERNEL_SHAPE+[len(INPUT)], OPTIMIZER, LOSS, METRICS)

model.fit(
    x=training, 
    epochs=EPOCHS)
import os
try:
    os.mkdir("checkpoints")
except:
    print('folder checkpoints already exists')
tf.keras.models.save_model(model, "checkpoints/cnn_checkpoint.ckp")