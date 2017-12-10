#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import sys

import numpy as np
import tensorflow as tf

# Fluff for record features
def int64_feature (value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature (value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature (value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def create_features (feature, value, pi_value, pi_index):
    return tf.train.Features (feature={
        'feature' : bytes_feature([feature.tostring()]),
        'value' : float_feature([value]),
        'pi_value' : float_feature(pi_value),
        'pi_index' : int64_feature(pi_index)
    })

def convert_example (feature, value, pi_value, pi_index):
    return tf.train.Example (
        features=create_features(feature, value, pi_value, pi_index)
    )

def write_dataset (directory, path, features, values, pi_values, pi_indices):
    path = os.path.join(directory, path)

    num_examples = len(features)

    if len(policies) != num_examples or len(values) != num_examples:
        print('Mismatch in number of samples', file=sys.stderr)
        exit(1)

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    with tf.python_io.TFRecordWriter(path, options=options) as writer:
        print('opened binary writer at {}'.format(path))

        for i in xrange(num_examples):
            feature = features[i]
            value = values[i]
            pi_value = pi_values[i]
            pi_index = pi_indices[i]

            example = convert_example(feature, value, pi_value, pi_index)
            writer.write(example.SerializeToString())

            if (i+1) % (num_examples // 100) == 0:
                print('{}:{}'.format(i+1, num_examples))

    print('closed binary writer at {}'.format(path))
