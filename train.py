#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import glob

import tensorflow as tf

import model
import model_fn
import input_fn

def main (FLAGS, _):
    builder = model.ModelSpecBuilder (
        model_fn=model_fn.model_fn,
        model_dir=FLAGS.model_dir
    )

    training_spec = builder.build_training_spec (
        input_fn=input_fn.dataset_input_fn (
            path=glob.glob(FLAGS.data_dir + '/*.tfrecords'),

            batch_size=FLAGS.batch_size,
            num_epochs=FLAGS.num_epochs,
            buffer_size=65535
        ),

        params={
            'filters' : FLAGS.filters,
            'modules' : FLAGS.modules,
            'n_classes' : FLAGS.n_classes,
            'optimizer' : FLAGS.optimizer,
            'learning_rate' : FLAGS.learning_rate,
            'l2_scale' : FLAGS.l2_scale
        }
    )

    training = model.TrainingModel(training_spec)

    with training:
        while not training.should_stop():
            loss = training.train()
            print('loss = {}'.format(loss))

    return 0

if __name__ == '__main__':
    import config
    import sys

    tf.logging.set_verbosity(tf.logging.INFO)

    FLAGS, unknown = config.get_FLAGS(config.config)
    exit(main(FLAGS, [sys.argv[0]] + unknown))
