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
import self_play
import util

from config import config

def run_game (game):
    # Initialize memory
    actions = []
    policies = []
    indices = []

    # Run through game
    node = game
    board = chess.Board()
    while not node.is_end():
        next_node = node.variation(0)
        move = next_node.move

        # Get action taken and action list
        action = adapter.move_to_label_flat(move)
        legal_actions = map(adapter.move_to_label_flat, board.legal_moves)

        # Create one-hot probability vector
        index = legal_actions.index(action)
        probs = util.one_hot(index, len(legal_actions))

        assert board.is_legal(move)

        # TODO: Look at the validity of this in case of underpromotion
        board.push(move)
        node = next_node

        # Update memory
        actions.append(action)
        policies.append(probs)
        indices.append(legal_actions)

    # Get game winner
    winner, outcome = {
        '1/2-1/2' : (chess.WHITE, 0.0),
        '1-0' : (chess.WHITE, 1.0),
        '0-1' : (chess.BLACK, 1.0)
    }.get(game.headers['Result'], None)

    return actions, policies, indices, outcome, winner

def run_pgn (pgn_file, n_games, data_dir):
    games = 0

    while n_games == 0 or games < n_games:
        game = chess.pgn.read_game(pgn_file)
        game.headers['Counter'] = games
        name = '{White}-{Black}-{ECO}-{Date}-{Counter}'.format (
            **game.headers
        ).replace(' ', '_')

        # Loop exit condition
        if game is None:
            break

        # Run through game generating labels
        actions, policies, indices, outcome, winner = run_game(game)

        # Save labels to disk
        self_play.write_records(data_dir, name, actions, policies, indices, outcome, winner)

        # 
        games += 1

def main (FLAGS, _):
    # Parse pgn into registry of nodes
    with open(FLAGS.pgn_file, 'r') as in_file:
        run_pgn(in_file, FLAGS.n_games, FLAGS.data_path)

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
