#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import sys

import chess
import chess.pgn

import numpy as np
import tensorflow as tf

import adapter
import output_fn
import util

from config import config

OUTCOME_DRAW = 0
OUTCOME_WHITE = 1
OUTCOME_BLACK = 2

class NodeRegistry (object):
    def __init__ (self):
        self.nodes = {}

    def get_node (self, board):
        key = board.epd()

        if not self.nodes.has_key(key):
            self.nodes[key] = Node()

        return self.nodes[key]

class Node (object):
    __slots__ = [
        'counts', 'outcomes'
    ]

    def __init__ (self, counts=None, outcomes=None):
        self.counts = {} if counts is None else counts
        self.outcomes = [0, 0, 0] if outcomes is None else outcomes

    def __repr__ (self):
        return 'Node(counts={}, outcomes={})'.format(
            self.counts, self.outcomes
        )

    def __str__ (self):
        return repr(self)

    def visit (self, move, outcome):
        # Add move to counts
        if not self.counts.has_key(move):
            self.counts[move] = 0

        # Count edge
        self.counts[move] += 1

        # Count game outcome
        self.outcomes[outcome] += 1

def run_game (game, registry):
    # Get game winner
    outcome = {
        '1/2-1/2' : OUTCOME_DRAW,
        '1-0' : OUTCOME_WHITE,
        '0-1' : OUTCOME_BLACK
    }[game.headers['Result']]

    node = game
    board = chess.Board()
    while not node.is_end():
        next_node = node.variation(0)
        move = next_node.move

        state_node = registry.get_node(board)
        state_node.visit(next_node.san(), outcome)

        board.push(move)
        node = next_node

def pgn_to_tree (pgn_file, n_games):
    registry = NodeRegistry()

    games = 0
    prev_position = 0
    while True:
        game = chess.pgn.read_game(pgn_file)

        # Loop exit condition
        if game is None:
            break

        run_game(game, registry)
        games += 1

        if games % max(n_games//10, 100) == 0:
            positions = len(registry.nodes)
            diff = positions - prev_position
            prev_position = positions
            print('games parsed: {}, unique positions: {} (+{})'.format(games, positions, diff))

        if n_games and games >= n_games:
            break

    return registry

def output_registry (out_file, registry, T):
    num_examples = len(registry.nodes)

    for i, key in enumerate(registry.nodes):
        # Get node and set up state
        node = registry.nodes[key]
        board, _ = chess.Board.from_epd(key)

        # Get features from state
        feature = adapter.position_to_hwc(board)

        # Calculate move probabilities
        all_moves = { board.san(m) : 0 for m in board.legal_moves }
        all_moves.update(node.counts)

        moves, counts = zip(*all_moves.items())
        moves = map(board.parse_san, moves)

        pi_value = util.softcount(counts, T=T)
        pi_index = map(adapter.move_to_label_flat, moves)

        # Calculate value estimates
        outcomes = util.softcount(node.outcomes, T=T)
        value = outcomes[OUTCOME_WHITE] - outcomes[OUTCOME_BLACK]

        # TODO: Look at relative evaluation
        # Change value relative which side to play
        if board.turn == chess.BLACK:
            value = -value

        # Write to file
        example = output_fn.convert_example(feature, value, pi_value, pi_index)
        out_file.write(example.SerializeToString())

        if (i+1) % (num_examples//10) == 0:
            print('written example {}/{}'.format(i+1, num_examples))

def main (FLAGS, _):
    # Parse pgn into registry of nodes
    with open(FLAGS.pgn_file, 'r') as in_file:
        registry = pgn_to_tree(in_file, FLAGS.n_games)

    # Make directory for data if needed
    path = FLAGS.data_path
    dirs, filename = os.path.split(path)
    if not os.path.exists(dirs):
        print('making directories {}'.format(dirs))
        os.makedirs(dirs)

    # Write registry to disk
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    with tf.python_io.TFRecordWriter(path, options=options) as out_file:
        print('opened binary writer at {}'.format(path))
        output_registry(out_file, registry, FLAGS.temperature)

    print('closed binary writer at {}'.format(path))
    print('examples written {}'.format(len(registry.nodes)))

    return 0

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()

    parser.add_argument (
        '--pgn_file', type=str, metavar='pgn',
        required=True,
        help='PGN database to parse.'
    )

    parser.add_argument (
        '--data_path', type=str, metavar='dir',
        required=True,
        help='Where to store output data.'
    )

    parser.add_argument (
        '--n_games', type=int, metavar='n',
        default=0,
        help='Number of games to parse.'
    )

    parser.add_argument (
        '--temperature', type=float, metavar='T',
        default=1.0,
        help='Temperature to use in softcount.'
    )

    FLAGS, unknown = parser.parse_known_args()
    exit(main(FLAGS, [sys.argv[0]] + unknown))
