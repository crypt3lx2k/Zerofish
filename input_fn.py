#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import tensorflow as tf

from config import config

# Placeholder input function section
def placeholder_dict (names, shapes, dtypes):
    placeholders = {}

    for name, shape, dtype in zip(names, shapes, dtypes):
        placeholders[name] = tf.placeholder (
            dtype=dtype, shape=(None,) + shape, name=name
        )

    return placeholders

def placeholder_input_fn (
        feature_names, feature_shapes, feature_dtypes,
        label_names, label_shapes, label_dtypes
):
    def input_fn ():
        features = placeholder_dict (
            feature_names,
            feature_shapes,
            feature_dtypes
        )

        labels = placeholder_dict (
            label_names,
            label_shapes,
            label_dtypes
        )

        return features, labels

    return input_fn

# TFRecords input function section
def parse_fn (example):
    example_features = {
        'feature' : tf.FixedLenFeature([], tf.string),
        'value' : tf.FixedLenFeature([1], tf.float32),
        'pi_value' : tf.VarLenFeature(tf.float32),
        'pi_index' : tf.VarLenFeature(tf.int64)
    }

    example = tf.parse_single_example(example, features=example_features)

    image = tf.decode_raw(example['feature'], tf.int8)
    image = tf.reshape(image, config.input_shape, name='image_reshaped')

    pi_value = tf.sparse_tensor_to_dense(example['pi_value'])
    pi_index= tf.sparse_tensor_to_dense(example['pi_index'])

    policy = tf.sparse_to_dense (
        sparse_indices=pi_index,
        output_shape=[config.n_classes],
        sparse_values=pi_value,
        validate_indices=False
    )

    value = example['value']

    return image, value, policy

def dataset_input_fn (path, batch_size=32, num_epochs=1, buffer_size=2048):
    def input_fn ():
        dataset = tf.contrib.data.TFRecordDataset(path, compression_type='GZIP')

        dataset = dataset.map(parse_fn)
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        image, value, policy = iterator.get_next()

        return {'image' : image}, {'value' : value, 'policy' : policy}

    return input_fn
