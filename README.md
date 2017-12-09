# ZeroFish
An implementation of the AlphaZero algorithm for chess

Currently under construction.

Currently uses a completely different model than the one from the paper! This model has a very different layout than the one from the paper. Significantly reduced number of parameters in value and policy output heads. Completely different action space. Does not deal with under-promotions. Action space is absolute compared to the relative to moving piece action spaced used in the paper. 

# Short guide to the different files

adapter.py - Acts a layer between the python-chess package and numpy. Supposed to handle all traffic between chess.Board and chess.Move to our neural network data and labels.

config.py - Handles all default values and the argument parser.

game_state.py - Acts as a layer of abstraction for chess/adapter to provide a game state that only deals with action indices.

input_fn.py - Handles the different types of input formats for tensorflow (placeholders, tfrecords data-sets).

mcts.py - Handles the MCTS algorithm.

model_fn.py - Contains the definition of our inference/evaluation/training model.

model.py - Handles building inference/evaluation/training specifications and acts as a layer of abstraction for our model_fn. Large parts of this could be replaced with tf.Estimator when the API matures.

output_fn.py - Handles output for the tfrecords format.

pgn_to_records.py - WIP supposed to handle reading pgn files and outputting tfrecords.

self_play.py - Plays one game of self-play and outputs tfrecords and pgn files.

train.py - Trains by reading tfrecords files from a directory.

visualize_filters.py - Shows image representations of the learned filters.
