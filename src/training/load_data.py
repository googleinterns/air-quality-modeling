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
import os
import json
import argparse

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Read TFRecords")
    parser.add_argument('--params_path',type=str)
    parser.add_argument('--tfrecords_path',type=str)
    args = parser.pargse_args()

    params = json.load(open(args.params_path, 'r'))
    tfrecord_path = args.tfrecords_path
    tfrecords_files = [os.path.join(tfrecords_path,f) for f in os.listdir(tfrecords_path)]




def load_tfrecords(folder, features_dict, input_feats, output_feats, out_scale=lambda x:x,
                           batch_size=32 ,buffer_size=300, parallel_calls=8):
    files = [ os.path.join(folder,f) for f in os.listdir(folder)]
    def parse_tfrecord(example_proto):
        """The parsing function.
        Read a serialized example into the structure defined by FEATURES_DICT.
        """
        return tf.io.parse_single_example(example_proto, features_dict)

    def to_tuple(inputs):
        """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
        Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
        Args:
          inputs: A dictionary of tensors, keyed by feature name.
        Returns:
          A tuple of (inputs, outputs).
        """
        inputsList = [inputs.get(key) for key in input_feats]
        outputsList = [inputs.get(key) for key in output_feats]
        in_stacked = tf.stack(inputsList, axis=0)
        out_stacked = tf.stack(outputsList, axis=0)
        in_stacked = tf.transpose(in_stacked, [1, 2, 0])
        out_stacked = tf.transpose(out_stacked, [1, 2, 0])
        return in_stacked, out_stacked    
    def scale_output(ins, outs):
        return ins, out_scale(outs)



    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=parallel_calls)
    dataset = dataset.map(to_tuple, num_parallel_calls=parallel_calls)
    dataset = dataset.map(scale_output, num_parallel_calls=parallel_calls)
    dataset = dataset.shuffle(buffer_size).batch(batch_size).repeat()
    return dataset

