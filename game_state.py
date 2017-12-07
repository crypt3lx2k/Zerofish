#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import adapter
import chess

class GameState (object):
    def __init__ (self, board=None):
        if board is None:
            self.state = chess.Board()
        else:
            self.state = board.copy()

        self.invalidate()

    def copy (self):
        return GameState(self.state)

    def invalidate (self):
        self._actions = None
        return self

    def actions (self):
        if self._actions is None:
            self._actions = map(adapter.move_to_label_flat, self.state.legal_moves)

        return self._actions

    def captures_or_evasions (self):
        moves = list(self.state.legal_moves)

        if not self.state.is_check():
            moves = filter(lambda m : self.state.is_capture(m), moves)

        return map(adapter.move_to_label_flat, moves)

    def parse_action (self, action):
        move = adapter.label_flat_to_move(action)

        # TODO: Fix this promotion stuff
        moving_piece = self.state.piece_at(move.from_square)
        if moving_piece.piece_type == chess.PAWN:
            rank, _ = adapter.square_to_index(move.to_square)

            # If pawn moves to edge promote to queen
            if rank == 0 or rank == 7:
                move.promotion = chess.QUEEN

        return move

    def push_action (self, action):
        move = self.parse_action(action)

        self.state.push(move)

        self.invalidate()
        return action

    def pop_action (self):
        move = self.state.pop()

        self.invalidate()
        return adapter.move_to_label_flat(move)

    def observation (self):
        return adapter.position_to_hwc(self.state)

    def done (self):
        return self.state.is_game_over()

    def reward (self):
        if self.state.is_checkmate():
            return -1.0

        return 0.0

    def turn (self):
        return self.state.turn
