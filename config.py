#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import argparse
import numpy as np

import util

config = util.AttrDict (**{
    # Default values for command line arguments
    'data_dir' : '/tmp/zf/data',
    'pgn_dir' : '/tmp/zf/pgns',
    'model_dir' : '/tmp/zf/model',

    'optimizer' : 'Adam',
    'learning_rate' : 1e-3,
    'l2_scale' : 1e-3,
    'num_epochs' : 1,
    'batch_size' : 32,

    'filters' : 32,
    'modules' : 0,

    # Not parsed from command line
    'input_height' : 8,
    'input_width' : 8,
    'input_channels' : 26,

    'classes_shape' : (8, 8, 8, 8),

    # Defined below
    'input_shape' : None,
    'input_total' : None,

    'n_classes' : None
})

config.input_shape = (config.input_height, config.input_width, config.input_channels)
config.input_total = np.prod(config.input_shape)
config.n_classes = np.prod(config.classes_shape)

def get_FLAGS (config):
    parser = argparse.ArgumentParser()

    parser.add_argument (
        '--data_dir', type=str, metavar='dir',
        default=config.data_dir,
        help='Directory where data is stored.'
    )

    parser.add_argument (
        '--pgn_dir', type=str, metavar='dir',
        default=config.pgn_dir,
        help='Directory where pgns are stored.'
    )

    parser.add_argument (
        '--model_dir', type=str, metavar='dir',
        default=config.model_dir,
        help='Directory to store model files.'
    )

    parser.add_argument (
        '--optimizer', type=str, metavar='opt',
        default=config.optimizer,
        help='Which optimizer to use for training.'
    )

    parser.add_argument (
        '--learning_rate', type=float, metavar='lr',
        default=config.learning_rate,
        help='Learning rate to use for training.'
    )

    parser.add_argument (
        '--l2_scale', type=float, metavar='x',
        default=config.l2_scale,
        help='Scale to use for l2 regularization.'
    )

    parser.add_argument (
        '--num_epochs', type=int, metavar='n',
        default=config.num_epochs,
        help='Number of epochs for training.'
    )

    parser.add_argument (
        '--batch_size', type=int, metavar='n',
        default=config.batch_size,
        help='Mini-batch size for training.'
    )

    parser.add_argument (
        '--filters', type=int, metavar='n',
        default=config.filters,
        help='Number of filters to use conv layers.'
    )

    parser.add_argument (
        '--modules', type=int, metavar='n',
        default=config.modules,
        help='Number of residual modules to use in model.'
    )

    # Get arguments from command line
    FLAGS, unparsed = parser.parse_known_args()

    # Overwrite config values with command line arguments
    config.update(FLAGS.__dict__)

    # Return modified config as FLAGS
    return config, unparsed
