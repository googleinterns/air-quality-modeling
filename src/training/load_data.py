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


import tensorflow as tf
import random

#timezone = TimezoneFinder(in_memory=True)


def load_tfrecords(files, features_dict, input_bands, output_bands,
                   parallel_calls=8):
    def parse_tfrecord(example_proto):
        """The parsing function.
        Read a serialized example into the structure defined by FEATURES_DICT.
        """
        return tf.io.parse_single_example(example_proto, features_dict)

    def stack_inputs(inputs):
        """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
        Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
        Args:
          inputs: A dictionary of tensors, keyed by feature name.
        Returns:
          A tuple of (inputs, outputs).
        """
        inputs_list = []
        for bands in input_bands:
            inputs_list.append(tf.stack([inputs.get(band) for band in bands],
                                        axis=-1))
        outputs = tf.stack([inputs.get(band) for band in output_bands], axis=-1)
        return inputs_list, outputs

    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=parallel_calls)
    dataset = dataset.map(stack_inputs, num_parallel_calls=parallel_calls)
    return dataset

def merge_features_without_date(inputs):
    return tf.stack(inputs[0][:-1],-1), inputs[1]

def augment_after_merge(inputs):
    return tf.image.rot90(inputs[0], k=random.randint(0, 4)), inputs[1]