#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import tempfile
import tensorflow as tf

cpu_config = tf.ConfigProto (
    device_count = {
        'CPU' : 1,
        'GPU' : 0
    },
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1,
)

class Model (object):
    def __init__ (self, spec):
        self.spec = spec
        self.contexts = []

    def __repr__ (self):
        return repr(self.spec)

    def __str__ (self):
        return repr(self.spec)

    def __enter__ (self):
        # Add context managers
        self.contexts.append(self.spec.graph.as_default())
        self.contexts.append(self.spec.session)

        # Enter contexts in order
        ctx_values = map(lambda ctx : ctx.__enter__(), self.contexts)
        return self

    def __exit__ (self, ex_type, ex_val, ex_trace):
        suppress = False

        # Pop contexts in reverse order as they were entered
        while self.contexts:
            ctx = self.contexts.pop()
            # Exit context
            ctx_suppress = ctx.__exit__(ex_type, ex_val, ex_trace)

            # Suppress if any of our managers decided to handle the error
            suppress = suppress or ctx_suppress

        return suppress

class TrainingModel (Model):
    def train (self):
        loss, _ = self.spec.session.run (
            [self.spec.loss, self.spec.train_op]
        )

        return loss

    def should_stop (self):
        return self.spec.session.should_stop()

class FeedingModel (Model):
    pass

class FeedingInferenceModel (FeedingModel):
    def feature_dict (self, features):
        return {
            self.spec.features[k] : features[k] for k in features
        }

    def infer (self, features):
        return self.spec.session.run (
            self.spec.predictions,
            feed_dict=self.feature_dict(features)
        )

class FeedingTrainingModel (FeedingInferenceModel):
    def label_dict (self, labels):
        return {
            self.spec.labels[k] : labels[k] for k in labels
        }

    def train (self, features, labels):
        feed_dict = {}
        feed_dict.update (
            self.feature_dict(features),
            self.label_dict(labels)
        )

        return self.spec.session.run (
            self.spec.train_op,
            feed_dict=feed_dict
        )

class ModelSpecBuilder (object):
    def __init__ (self, model_fn=None, model_dir=None, config=None):
        self.model_fn = model_fn
        self.model_dir = model_dir
        self.config = config

        if self.model_dir is None:
            self.model_dir = tempfile.mkdtemp()
            tf.logging.warning('Using {} as model_dir'.format(self.model_dir))

    def build_graph (self, input_fn, mode, params, graph=None):
        if graph is None:
            graph = tf.Graph()

        with graph.as_default():
            global_step = tf.train.get_or_create_global_step()

            with tf.variable_scope('input'):
                features, labels = input_fn()

            spec = self.model_fn (
                features=features,
                labels=labels,
                mode=mode,
                params=params
            )

            spec.global_step = global_step

        spec.graph = graph
        return spec

    def build_inference_spec (self, input_fn, params, graph=None):
        checkpoint_path = tf.train.latest_checkpoint(self.model_dir)

        if not checkpoint_path:
            tf.logging.warning('Found no trained model in {}'.format(self.model_dir))

        spec = self.build_graph (
            input_fn=input_fn,
            mode=tf.estimator.ModeKeys.PREDICT,
            params=params,
            graph=graph
        )

        with spec.graph.as_default() as graph:
            global_step = tf.train.get_or_create_global_step()
            session = tf.train.MonitoredSession (
                session_creator=tf.train.ChiefSessionCreator(
                    checkpoint_dir=self.model_dir,
                    config=self.config
                )
            )

            spec.session = session

        spec.graph.finalize()
        return spec

    def build_training_spec (self, input_fn, params, graph=None):
        spec = self.build_graph (
            input_fn=input_fn,
            mode=tf.estimator.ModeKeys.TRAIN,
            params=params,
            graph=graph
        )

        with spec.graph.as_default() as graph:
            session = tf.train.MonitoredTrainingSession (
                checkpoint_dir=self.model_dir,
                config=self.config
            )

            spec.session = session

        return spec
