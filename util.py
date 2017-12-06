#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import numpy as np

class AttrDict (dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__ (self, **kwargs):
        super(AttrDict, self).__init__(**kwargs)

        # Apply AttrDict on each dictionary in self recursively
        for k in self:
            v = self[k]

            if isinstance(v, AttrDict):
                continue

            if isinstance(v, dict):
                self[k] = AttrDict(**v)

def one_hot (indices, shape, dtype=np.float32):
    """Returns one-hot encoded vector of given shape with 1 for each index in indices."""
    v = np.zeros(shape, dtype=dtype)
    v[indices] = dtype(1)
    return v

def softcount (counts, T=1.0):
    """Returns probability distribution given count vector."""
    if T == 0.0:
        return one_hot (
            indices=np.argmax(counts),
            shape=np.array(counts).shape,
            dtype=np.float32
        )

    counts = np.array(counts, dtype=np.float32)

    if T != 1.0:
        counts **= (1.0/T)

    total = counts.sum()
    counts /= total

    return counts

def softmax (logits, T=1.0):
    """Returns probability distribution given unscaled log probabilities."""
    if T == 0.0:
        return one_hot (
            indices=np.argmax(logits),
            shape=np.array(logits).shape,
            dtype=np.float32
        )

    if T != 1.0:
        logits /= T

    logits -= logits.max()
    e_l = np.exp(logits)

    return e_l/e_l.sum()

def inverse_map (iterable):
    """Returns an inverse mapping of iterable."""
    mapping = {}

    for i, e in enumerate(iterable):
        mapping[e] = i

    return mapping
