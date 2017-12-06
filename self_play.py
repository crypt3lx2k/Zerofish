#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import datetime
import os

import chess

import numpy as np
import tensorflow as tf

import adapter
import config
import game_state
import input_fn
import mcts
import model
import model_fn
import output_fn

def play_game (inference):
    # Initialize memory
    actions = []
    policies = []
    indices = []
    moves = []

    # Set up search tree
    state = game_state.GameState()
    tree = mcts.MCTS(inference, state)

    # Play game
    while not tree.state.done():
        print(tree.state.state.unicode())

        # Perform search
        node = tree.search(1024)

        # Calculate move probabilities and get action index
        probs = mcts.policy(node, T=1.0)
        index = np.random.choice(len(node.actions), p=probs)

        # Get action and update tree
        action = node.actions[index]
        value = node.Q[index]
        move = tree.state.parse_action(action)

        print(tree.state.state.san(move))

        tree.act(index)

        # Store stats
        actions.append(action)
        policies.append(probs)
        indices.append(node.actions)
        moves.append(move)

    # Get game outcome and last player to move
    outcome = tree.state.reward()
    last_turn = not tree.state.turn()

    print(tree.state.state.unicode())
    print(chess.Board().variation_san(moves))

    return actions, policies, indices, outcome, last_turn

def write_game_records (out_file, actions, policies, indices, outcome, last_turn):
    # Create new state
    state = game_state.GameState()
    moves = []

    # Run through game to create feature vectors and produce output
    for i, action in enumerate(actions):
        # Extract features
        feature = state.observation().get('image')

        # Calculate value of game based on who's to play
        value = outcome if state.turn() == last_turn else -outcome

        # Write example to disk
        example = output_fn.convert_example(feature, value, policies[i], indices[i])
        out_file.write(example.SerializeToString())

        # Update game state
        state.push_action(action)
        moves.append(state.state.peek())

    return moves

def main (FLAGS, _):
    builder = model.ModelSpecBuilder (
        model_fn=model_fn.model_fn,
        model_dir=FLAGS.model_dir
    )

    inference_spec = builder.build_inference_spec (
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

    inference = model.FeedingInferenceModel(inference_spec)

    # Make directory for data if needed
    dirs = FLAGS.data_dir
    if not os.path.exists(dirs):
        print('making directories {}'.format(dirs))
        os.makedirs(dirs)

    with inference:
        actions, policies, indices, outcome, last_turn = play_game(inference)

    # Create file path
    name = datetime.datetime.utcnow().isoformat()
    path = '{}/{}.{}'.format(dirs, name, 'tfrecords')

    # Open tfrecords file
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    with tf.python_io.TFRecordWriter(path, options=options) as out_file:
        print('opened binary writer at {}'.format(path))
        moves = write_game_records(out_file, actions, policies, indices, outcome, last_turn)
    print('closed binary writer at {}'.format(path))

    path = '{}/{}.{}'.format(dirs, name, 'pgn')
    pgn = [chess.Board().variation_san(moves)]

    if outcome:
        if last_turn == chess.WHITE:
            pgn.append('1-0')
        else:
            pgn.append('0-1')
    else:
        pgn.append('1/2-1/2')

    print('writing {}'.format(path))
    with open(path, 'w') as out_file:
        print(' '.join(pgn), file=out_file)

    return 0

if __name__ == '__main__':
    import config
    import sys

    tf.logging.set_verbosity(tf.logging.INFO)

    FLAGS, unknown = config.get_FLAGS(config.config)
    exit(main(FLAGS, [sys.argv[0]] + unknown))
