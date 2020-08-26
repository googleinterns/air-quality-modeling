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

import tensorflow as tf
import json
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--params_path', type=str)
    parser.add_argument('--tfrecords_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    args = parser.parse_args()

    args = parser.pargse_args()

    params = json.load(open(args.params_path, 'r'))
    tfrecords_path = args.tfrecords_path
    tfrecords_files = [os.path.join(tfrecords_path, f) for f in
                       os.listdir(tfrecords_path)]

    KERNEL_RADIUS = params['kernel_radius']
    KERNEL_SIZE = KERNEL_RADIUS*2+1
    KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
    BANDS = []
    for k in params['bands'].keys():
        BANDS = BANDS + params['bands'][k]
    PATCH_BANDS = ["patch_%s" % b for b in BANDS]
    BANDS = BANDS + ["HOW", "DOW", "DOM", "MOY", "latitude", "longitude"]
    FEATURES = PATCH_BANDS + BANDS
    COLUMNS = [
        tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32)
        for k in PATCH_BANDS] + [
                tf.io.FixedLenFeature(shape=[1, 1], dtype=tf.float32)
                for k in BANDS]
    FEATURES_DICT = dict(zip(FEATURES, COLUMNS))

    OUTPUT = ['troposhpheric_NO2_column_number_density']



EPOCHS = 1000
CKP_EPOCH = 5
BTACH_SIZE  = 32

#folder = "/usr/local/google/home/aghriss/airqual/datasets"
#dataset = load_tfrecords(folder, FEATURES_DICT, INPUT, OUTPUT)

OPTIMIZER = 'RMSprop'
LOSS = 'MeanSquaredError'
METRICS = ['RootMeanSquaredError']

import utils
from neurals import cnn

training = utils.load_tfrecords(FOLDER, FEATURES_DICT, INPUT, OUTPUT, out_scale= lambda x: 100*x-0.5)
model = cnn.get_model([None, None]+[len(INPUT)], OPTIMIZER, LOSS, METRICS)

import os
try:
    os.mkdir("checkpoints")
except:
    print('folder checkpoints already exists')

for i in range(EPOCHS):
    history = model.fit(
    x=training, 
    steps_per_epoch = 1000,
    epochs=1)
    if not (i+1)%CKP_EPOCH:

        tf.keras.models.save_model(model, "checkpoints/cnn_checkpoint_%i.ckp"%(i+1))
    #print(history)
        json.dump(history.history, open("checkpoints/history_cnn_%i.json"%(i+1),'w'))