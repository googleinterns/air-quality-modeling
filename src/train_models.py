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

import json
import os
import argparse
import training
import random
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics
from tensorflow.python.keras import optimizers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--params_path', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--tfrecords_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    args = parser.parse_args()

    args = parser.pargse_args()

    params = json.load(open(args.params_path, 'r'))
    model_type = args.model_type
    gpu_index = args.gpu_index
    tfrecords_path = args.tfrecords_path
    checkpoint_path = args.checkpoint_path

    assert gpu_index in [0, 1], "Index should be either 0 or 1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    import tensorflow as tf

    kernel_radius = params['kernel_radius']
    tfrecord_files = [os.path.join(tfrecords_path, f) for f in
                      os.listdir(tfrecords_path)]
    random.shuffle(tfrecord_files)

    patch_size = [2 * kernel_radius + 1] * 2
    spectral_bands = ["patch_%s" % b for b in ['B', 'G', 'R', 'N', 'P']]
    tropo_bands = ['patch_cloud_fraction']
    wind_bands =["patch_%s"%b for b in ['11_u_component_of_wind_10m', '11_v_component_of_wind_10m', '10_u_component_of_wind_10m', '10_v_component_of_wind_10m',
             '9_u_component_of_wind_10m', '9_v_component_of_wind_10m', '8_u_component_of_wind_10m', '8_v_component_of_wind_10m',
             '7_u_component_of_wind_10m', '7_v_component_of_wind_10m', '6_u_component_of_wind_10m', '6_v_component_of_wind_10m',
             '5_u_component_of_wind_10m', '5_v_component_of_wind_10m', '4_u_component_of_wind_10m', '4_v_component_of_wind_10m', 
             '3_u_component_of_wind_10m', '3_v_component_of_wind_10m', '2_u_component_of_wind_10m', '2_v_component_of_wind_10m',
             '1_u_component_of_wind_10m', '1_v_component_of_wind_10m', '0_u_component_of_wind_10m', '0_v_component_of_wind_10m']]
    dsm_bands = ['patch_dsm']
    road_bands = ['patch_num_observations']
    patch_bands = (spectral_bands + tropo_bands + dsm_bands + wind_bands +
                   road_bands)

    date_bands = ['longitude', 'latitude', 'HOD', 'DOW', 'DOM', 'MOY']
    output = ['tropospheric_NO2_column_number_density']

    columns = [tf.io.FixedLenFeature(shape=patch_size, dtype=tf.float32)
               for k in patch_bands] + [
               tf.io.FixedLenFeature(shape=[1, 1], dtype=tf.float32)
               for k in date_bands] + [
               tf.io.FixedLenFeature(shape=[1, 1], dtype=tf.float32)]
    features = patch_bands + date_bands + output
    features_dict = dict(zip(features, columns))
    input_bands = [spectral_bands, tropo_bands, dsm_bands, wind_bands,
                   road_bands, date_bands]

    train_dataset = training.load_data(tfrecord_files[:-10], features_dict,
                                       input_bands, output)
    eval_dataset = training.load_data(tfrecord_files[-10:], features_dict,
                                      input_bands, output)

    if model_type.upper() == "CNN":
        number_of_bands = sum([len(bands) for bands in input_bands])
        inputs = layers.Input(shape=[None, None, number_of_bands])
        train_dataset = train_dataset.map(training.merge_features_without_date)
        train_dataset = train_dataset.map(training.augment_after_merge)
        eval_dataset = eval_dataset.map(training.merge_features_without_date)
        eval_dataset = eval_dataset.map(training.augment_after_merge)
        model = training.get_cnn_model(inputs)
    model.summary()
    EPOCHS = 20
    BUFFER_SIZE = 1000
    BATCH_SIZE = 32
    CKP_EPOCH = 5
    OPTIMIZER = 'RMSprop'
    LOSS = 'MeanSquaredError'
    METRICS = ['RootMeanSquaredError']

    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    eval_dataset = eval_dataset.batch(1).repeat()

    model.compile(optimizer=optimizers.get(OPTIMIZER), loss=losses.get(LOSS),
                  metrics=[metrics.get(metric) for metric in METRICS])

    try:
        os.mkdir(checkpoint_path)
    except FileExistsError:
        print('folder checkpoints already exists')

    for i in range(EPOCHS):
        history = model.fit(x=training_dataset, steps_per_epoch=1000,
                            epochs=1, validation_data = eval_dataset,
                            validation_steps = 1000)
        if (i+1)%CKP_EPOCH != 0:
            tf.keras.models.save_model(model,
                                       os.path.join(checkpoint_path,
                                                    model_type+"_%i.ckp" % i))
            json.dump(history.history,
                      open(os.path.join(checkpoint_path, model_type +
                                        "_%i.json" % i), 'w'))
