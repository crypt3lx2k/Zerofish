#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import chess
import numpy as np

from config import config

# Move label routines
def square_to_index (square):
    """Maps square to our 2d-index format."""
    return np.unravel_index(square, config.input_shape[0:2])

def move_to_label (move):
    """Maps move to our 4d-label format."""
    sq_f = move.from_square
    sq_t = move.to_square

    label = (
        square_to_index(sq_f) +
        square_to_index(sq_t)
    )

    return label

def move_to_label_flat (move):
    """Maps move to flat 1d-label."""
    return np.ravel_multi_index(
        move_to_label(move),
        config.classes_shape
    )

def index_to_square (idx):
    """Maps from our 2d-index format to square."""
    return chess.square(idx[1], idx[0])

def label_to_move (label):
    """Maps from our 4d-label format to a move."""
    return chess.Move (
        index_to_square(label[0:2]),
        index_to_square(label[2:4])
    )

def label_flat_to_move (label):
    """Maps from flat 1d-label to move."""
    return label_to_move(
        np.unravel_index(label, config.classes_shape)
    )

# Bitboard routines
def squares_to_bb (squares):
    """Maps a list of squares to the equivalent 8x8 `bitboard'."""
    indices = map(square_to_index, squares)

    bb = np.zeros(config.input_shape[0:2], dtype=np.byte)
    bb[zip(*indices)] = 1

    return bb

def position_to_legal_bbs (position):
    """Returns the legal moves as an (8x8x8x8) binary vector matching our 4d-label format."""
    legal_moves = list(position.legal_moves)
    indices = map(move_to_label, legal_moves)

    bbs = np.zeros(config.classes_shape, dtype=np.byte)
    bbs[zip(*indices)] = 1

    return bbs

def bool_to_bb (boolean):
    """Maps boolean to either empty or full bitboard."""
    # TODO: Cache these? Only two possible output values.
    allocator = np.ones if boolean else np.zeros
    bb = allocator(config.input_shape[0:2], dtype=np.byte)

    return bb

def position_to_bool_bbs (position):
    """Returns all the relevant boolean bitboards from position."""
    colors = chess.COLORS

    to_play = [position.turn]
    is_check = [position.is_check()]
    kingside = map(position.has_kingside_castling_rights, colors)
    queenside = map(position.has_queenside_castling_rights, colors)

    bools = (
        to_play +
        is_check +
        kingside +
        queenside
    )

    bool_bbs = map(bool_to_bb, bools)

    return np.array(bool_bbs)

def position_to_chw (position):
    """Returns all model input feature planes from position in chw format."""
    # Piece boards
    occupancies = [
        position.pawns,   position.knights,
        position.bishops, position.rooks,
        position.queens,  position.kings
    ]

    # Full piece boards for each color
    occupancies = map(chess.SquareSet, occupancies)
    colors = map(chess.SquareSet, position.occupied_co)

    # Color specific piece boards
    occupancies_black = map(lambda bb : bb & colors[chess.BLACK], occupancies)
    occupancies_white = map(lambda bb : bb & colors[chess.WHITE], occupancies)

    squaresets = (
        occupancies +
        colors +
        occupancies_black +
        occupancies_white
    )

    squares = map(list, squaresets)
    bbs = map(squares_to_bb, squares)
    bbs = np.array(bbs, dtype=np.byte)

    bool_bbs = position_to_bool_bbs(position)

    bbs_chw = np.concatenate (
        [
            bbs,
            bool_bbs
        ],
        axis = 0
    )

    return bbs_chw

def position_to_hwc (position):
    """Returns all model input feature planes from position in hwc format."""
    bbs_chw = position_to_chw(position) 

    # chw -> hwc
    return bbs_chw.transpose((1, 2, 0))
