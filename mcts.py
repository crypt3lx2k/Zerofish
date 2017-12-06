#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import numpy as np

import util

# TODO: Fix these variables
c_puct = 5.0
eps = 0.25
dirichlet_alpha = 0.3
n_vl = 5

def policy (node, T=0.0):
    return util.softcount(node.N, T=T)

class SingleThreadedCache (object):
    def __init__ (self, model, state):
        self.model = model
        self.state = state

        self.cache = {}

    def key (self):
        return self.state.state.epd()

    def infer (self, observation):
        key = self.key()

        if not self.cache.has_key(key):
            logits, value = self.model.infer (
                observation
            )

            logits = logits[0]
            value = value[0][0]

            self.cache[key] = (logits, value)

        return self.cache[key]

class Node (object):
    __slots__ = ['actions', 'children', 'N', 'Q', 'P']

    def __init__ (self, actions):
        num_actions = len(actions)
        
        # Tree pointers
        self.actions = actions
        self.children = {}

        # Node statistics
        self.N = np.zeros(num_actions, dtype=np.int32)
        self.Q = np.zeros(num_actions, dtype=np.float32)
        self.P = np.zeros(num_actions, dtype=np.float32)

    def first_zero_index (self):
        return np.argmin(self.N)

    def puct_index (self):
        # If any N are zero, expand node before PUCT
        if np.any(self.N == 0):
            return self.first_zero_index()

        # Calculate action values and upper confidence bound
        Q = self.Q
        U = c_puct * self.P * np.sqrt(self.N.sum())/(1 + self.N)

        # Get index that maximizes sum of Q estimate and upper confidence bound
        return np.argmax(Q+U)

    def get_child (self, index):
        return self.children.get(index, None)

    def set_child (self, index, child):
        self.children[index] = child
        return child

    def terminal (self):
        return len(self.actions) == 0

    def select (self, index):        
        # Update node statistics
        N = self.N[index]
        Q = self.Q[index]

        W = N*Q

        self.N[index] = N + n_vl
        self.Q[index] = (W - n_vl)/(N + n_vl)

    def backup (self, index, value):
        N = self.N[index]
        Q = self.Q[index]

        W = N*Q

        self.N[index] = N - n_vl + 1
        self.Q[index] = (W + n_vl + value)/(N - n_vl + 1)

    def pv (self):
        return self.pv_recurse([])

    def pv_recurse (self, moves):
        if np.all(self.N == 0):
            return moves

        index = np.argmax(self.N)
        action = self.actions[index]

        moves.append(action)
        return self.children[index].pv_recurse(moves)

class MCTS (object):
    def __init__ (self, model, state):
        self.model = SingleThreadedCache(model, state)
        self.state = state

        self.root, self.root_value = self.expand()

    def search (self, num_simulations):
        for simulation in xrange(num_simulations):
            self.search_root()

        return self.root

    def act (self, index):
        # Get action from node
        action = self.root.actions[index]

        # Look up new node and value
        child = self.root.get_child(index)
        value = self.root.Q[index]

        # Update root with child node
        self.root = child
        self.root_value = value

        # Update game state with action
        self.state.push_action(action)

        # Early exit if game is over
        if self.state.done():
            return self

        # If child node didn't exist we need to expand root 
        if child is None:
            self.root, self.root_value = self.expand()

        return self

    def search_root (self):
        # Set current node to root node
        node = self.root

        # Keep track of path taken
        trajectory = []

        # Find new action to expand
        while True:
            child, index = self.select(node)
            trajectory.append((node, index))

            if child is None:
                break

            node = child

        # Back up value if terminal
        if self.state.done():
            return self.backup(trajectory, self.state.reward())

        # Create new node
        child, value = self.expand()

        # Link child into tree
        node.set_child(index, child)

        # Back up value-estimate to root
        return self.backup(trajectory, value)

    def select (self, node):
        # We're in a valid node so pick index
        index = node.puct_index()
        child = node.get_child(index)

        # Update statistics
        node.select(index)

        # Find action and update state
        action = node.actions[index]
        self.state.push_action(action)

        return child, index

    def backup (self, trajectory, value):
        while trajectory:
            action = self.state.pop_action()

            node, index = trajectory.pop()
            value = -value

            node.backup(index, value)

            assert node.actions[index] == action

    def expand (self):
        # Neural network evaluation
        logits, value = self.model.infer (
            self.state.observation()
        )

        # Legal actions in this state
        actions = sorted (
            self.state.actions(),
            key=lambda action : logits[action],
            reverse=True
        )

        # Prior probabilities
        logits = logits[actions]
        probs = util.softmax(logits, T=1.0)

        # Dirichlet noise to priors
        alpha = np.zeros_like(probs)
        alpha[:] = dirichlet_alpha
 
        noise = np.random.dirichlet(alpha)
        probs = (1-eps)*probs + eps*noise

        # Make new node for this position
        node = Node(actions)
        node.P[:] = probs

        return node, value

