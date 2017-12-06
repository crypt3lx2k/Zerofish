#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import numpy as np
import tensorflow as tf

import util

# Temporary workaround for tf1.3 and down
def collection_getter (getter, *args, **kwargs):
    """Adds variables to relevant collections."""
    var = getter(*args, **kwargs)

    name = kwargs['name']
    trainable = kwargs['trainable']

    if trainable:
        if 'kernel' in name:
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, var)

        if 'bias' in name:
            tf.add_to_collection(tf.GraphKeys.BIASES, var)

    return var

def input_module (net, filters, training=False):
    """First layer that interfaces directly with binary feature vectors."""

    # 7x7 -> batch norm -> relu
    net = tf.layers.conv2d(net, filters, (7, 7), (1, 1), padding='SAME', name='conv_7x7/conv')
    net = tf.layers.batch_normalization(net, training=training, name='conv_7x7/norm')
    net = tf.nn.relu(net, name='conv_7x7/relu')

    return net

def residual_module (net, training=False):
    """Residual module, uses two 3x3 convs and post-activation."""
    filters = int(net.shape.as_list()[-1])

    # Along this branch 3x3 -> batch norm -> relu -> 3x3 -> batch norm
    with tf.variable_scope('branch'):
        branch = net
        branch = tf.layers.conv2d (
            branch, filters, (3, 3), (1, 1), padding='SAME', name='conv_3x3_0/conv'
        )
        branch = tf.layers.batch_normalization(branch, training=training, name='conv_3x3_0/norm')
        branch = tf.nn.relu(branch, name='conv_3x3_0/relu')

        branch = tf.layers.conv2d (
            branch, filters, (3, 3), (1, 1), padding='SAME', name='conv_3x3_1/conv'
        )
        branch = tf.layers.batch_normalization(branch, training=training, name='conv_3x3_1/norm')

    # Residual head, original input + branch -> batch norm -> relu
    with tf.variable_scope('residual'):
        net += branch
        net = tf.layers.batch_normalization(net, training=training, name='norm')
        net = tf.nn.relu(net, name='relu')

    return net

def output_module (net, filters, training=False):
    """Final shared layer before two headed output."""

    # 1x1 -> batch norm -> relu
    net = tf.layers.conv2d (
        net, filters, (1, 1), (1, 1), padding='SAME', name='conv_1x1/conv'
    )
    net = tf.layers.batch_normalization(net, training=training, name='conv_1x1/norm')
    net = tf.nn.relu(net, name='conv_1x1/relu')

    return net

def output_policy (net, n_classes, training=False):
    """Policy head for the network."""
    branch = net

    # 1x1 -> flattening
    branch = tf.layers.conv2d (
        branch, n_classes//(8*8), (1, 1), (1, 1), padding='SAME', name='logits'
    )

    dims = np.prod(branch.shape.as_list()[1:])
    branch = tf.reshape(branch, [-1, dims], name='flattened_logits')

    return branch

def output_value (net, training=False):
    """Value head for the network."""
    branch = net

    # flattening -> dense of scalar output
    dims = np.prod(branch.shape.as_list()[1:])
    branch = tf.reshape(branch, [-1, dims], name='flattened')

    branch = tf.layers.dense(branch, 1, name='logits')
    branch = tf.nn.tanh(branch, name='logits/tanh')

    return branch

def inference (inputs, filters, modules, n_classes, training=False):
    """Complete inference model."""
    net = inputs

    # Cast binary features into float32 (-> batch norm)
    with tf.variable_scope('transform'):
        net = tf.cast(net, tf.float32, name='image_float')
        net = tf.layers.batch_normalization(net, training=training, name='image_float/norm')

    # Run binary features through input module
    with tf.variable_scope('input'):
        net = input_module(net, filters, training=training)

    # Some amount of residual modules
    for layer in xrange(modules):
        with tf.variable_scope('residual_{}'.format(layer)):
            net = residual_module(net, training=training)

    # Output layer and policy, value heads
    with tf.variable_scope('output'):
        net = output_module(net, 8*8, training=training)

        with tf.variable_scope('policy'):
            policy = output_policy(net, n_classes, training=training)

        with tf.variable_scope('value'):
            value = output_value(net, training=training)

    return policy, value

def model_fn (features, labels, mode, params):
    # Training flag
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Extract and concatenate features for input
    inputs = features['image']

    # Get unscaled log probabilities
    with tf.variable_scope (
            'inference',
            reuse=params.get('reuse', False),
            custom_getter=collection_getter
    ):
        policy, value = inference (
            inputs,
            filters=params['filters'],
            modules=params['modules'],
            n_classes=params['n_classes'],
            training=training
        )

    # Add summaries to weights
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name.split(':')[0] + '_summary', var)

    # Specification
    spec = util.AttrDict (
        mode=mode,
        features=features,
        predictions=(policy, value)
    )

    # Return early inference specification
    if mode == tf.estimator.ModeKeys.PREDICT:
        return spec

    with tf.variable_scope('losses'):
        # Value loss
        value_loss = tf.losses.mean_squared_error (
            labels=labels['value'], predictions=value,
            weights=1.0/4.0
        )

        policy_loss = tf.losses.softmax_cross_entropy (
            onehot_labels=labels['policy'], logits=policy,
            weights=1.0
        )

        # Get l2 regularization loss
        l2_loss = tf.contrib.layers.apply_regularization (
            tf.contrib.layers.l2_regularizer(params['l2_scale'])
        )
        tf.losses.add_loss(l2_loss)

        # Total loss
        loss = tf.losses.get_total_loss(add_regularization_losses=False)

        # Add total loss to loss collection
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

    # Add summaries for losses
    for loss_tensor in tf.get_collection(tf.GraphKeys.LOSSES):
        tf.summary.scalar(loss_tensor.name.split(':')[0] + '_summary', loss_tensor)

    spec.labels = labels
    spec.loss = loss
    spec.eval_metric_ops = util.AttrDict()

    # Return early evaluation specification
    if mode == tf.estimator.ModeKeys.EVAL:
        return spec

    # Get global step for training op
    global_step = tf.train.get_global_step()

    with tf.variable_scope('train'):
        # Get optimizer function
        optimizer_fn = {
            'Adam' : tf.train.AdamOptimizer,
            'RMSProp' : tf.train.RMSPropOptimizer,
            'GradientDescent' : tf.train.GradientDescentOptimizer
        }[params.get('optimizer', 'Adam')]

        optimizer = optimizer_fn(params['learning_rate'])

        # Compute gradients and add summaries
        grads_and_tvars = optimizer.compute_gradients(spec.loss)

        # Create train operation
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients (
                grads_and_tvars,
                global_step=global_step
            )

    # Add summaries for gradients
    with tf.variable_scope('gradients'):
        tf.contrib.training.add_gradients_summaries(grads_and_tvars)

    spec.train_op = train_op

    # Return full train specification
    return spec
