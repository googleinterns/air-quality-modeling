import tensorflow as tf
import os


def load_tfrecords(folder, features_dict, input_feats, output_feats,
                           batch_size=32,buffer_size=300):
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

    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
    dataset = dataset.map(to_tuple, num_parallel_calls=5)
    dataset = dataset.shuffle(buffer_size).batch(batch_size).repeat()
    return dataset

