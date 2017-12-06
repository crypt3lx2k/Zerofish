#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import sys

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import adapter
import model
import model_fn
import input_fn

def visualize_filter (x, y):
    # Subtract biases to get proper value scale for each output channel
    x -= y/x.shape[-1]

    # Normalize to [0, 1]
    x_min = x.min()
    x_max = x.max()

    x = (x-x_min)/(x_max-x_min)

    # HWCN -> CNHW
    x = np.transpose(x, [2, 3, 0, 1])

    # Pad in between filter images
    m, n, h, w = x.shape

    x = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')

    x = np.transpose(x, [0, 2, 1, 3])
    x = np.reshape(x, [(h+2)*m, (w+2)*n])

    # Show images to screen
    plt.imshow(x, cmap='hot')

def name_filter (name, suffix):
    return (('conv' in name) or ('logits' in name)) and suffix in name

def main (FLAGS, _):
    # Initialize all variables
    if not os.path.exists(FLAGS.model_dir):
        print('No model at {}'.format(FLAGS.model_dir), file=sys.stderr)
        return 1

    builder = model.ModelSpecBuilder (
        model_fn=model_fn.model_fn,
        model_dir=FLAGS.model_dir
    )

    spec = builder.build_inference_spec (
        input_fn=input_fn.placeholder_input_fn (
            feature_names=('image',),
            feature_shapes=(FLAGS.input_shape,),
            feature_dtypes=(tf.int8,),

            label_names=('policy', 'value'),
            label_shapes=((FLAGS.n_classes,), (1,)),
            label_dtypes=(tf.float32, tf.float32),
        ),

        params={
            'filters' : FLAGS.filters,
            'modules' : FLAGS.modules,
            'n_classes' : FLAGS.n_classes
        }
    )

    # Get all weights of action value network
    tvars = spec.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # Extract only the convolutional filters
    filters = filter(lambda v : name_filter(v.name, 'kernel'), tvars)
    biases = filter(lambda v : name_filter(v.name, 'bias'), tvars)
    names = map(lambda v : v.name, filters)

    filters, biases = spec.session.run([filters, biases])

    i = 0
    for a, b in zip(filters, biases):
        print(names[i])
        i += 1
        if a.ndim < 4:
            a.resize((1, 1, 64, 64))
        visualize_filter(a, b)
        plt.show()
        # # Show only first
        # break

    return 0

if __name__ == '__main__':
    import config

    tf.logging.set_verbosity(tf.logging.INFO)

    FLAGS, unparsed = config.get_FLAGS(config.config)
    exit(main(FLAGS, [sys.argv[0]] + unparsed))
