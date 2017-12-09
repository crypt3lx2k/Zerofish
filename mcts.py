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
        # # If any N are zero, expand node before PUCT
        # if np.any(self.N == 0):
        #     return self.first_zero_index()

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

class VirtualThread (object):
    def __init__ (self, parent):
        self.parent = parent
        self.state = self.parent.state.copy()
        
        self.node = None
        self.index = None
        self.trajectory = None

    def get (self, node):
        # Keep track of path taken
        trajectory = []

        # Find new action to expand
        while True:
            child, index = self.select(node)
            trajectory.append((node, index))

            if child is None:
                break

            node = child

        # Store search state
        self.node = node
        self.index = index
        self.trajectory = trajectory

        return self.state

    def put (self, logits, value):
        # Get search state
        node = self.node
        index = self.index
        trajectory = self.trajectory

        # Back up value if terminal
        if self.state.done():
            return self.backup(trajectory, self.state.reward())

        # If child node doesn't exist
        if node.get_child(index) is None:
            # Create child
            child = self.parent.expand(self.state.actions(), logits)

            # Link child into tree
            node.set_child(index, child)

        # Back up value-estimate to root
        return self.backup(trajectory, value)

    def push_action (self, action):
        self.state.push_action(action)

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

class MCTS (object):
    def __init__ (self, model, state, num_threads=1):
        self.model = model
        self.state = state

        self.num_threads = num_threads
        self.threads = self.spawn_threads()

        self.root = self.expand_root()

    def spawn_threads (self):
        threads = []

        for thread_id in xrange(self.num_threads):
            thread = VirtualThread(self)
            threads.append(thread)

        return threads

    def search (self, num_simulations):
        for simulation in xrange(num_simulations//self.num_threads):
            states = []

            for thread in self.threads:
                state = thread.get(self.root)
                states.append(state)

            observations = map(lambda state : state.observation(), states)
            observations = np.stack(observations)
            batch_logits, batch_values = self.model.infer({'image' : observations})

            for i, thread in enumerate(self.threads):
                thread.put(batch_logits[i], batch_values[i][0])

        return self.root

    def act (self, index):
        # Get action from node
        action = self.root.actions[index]

        # Look up new node and value
        child = self.root.get_child(index)
        value = self.root.Q[index]

        # Update root with child node
        self.root = child

        # Update game state with action
        self.state.push_action(action)

        # Push action to threads
        for thread in self.threads:
            thread.push_action(action)

        # Early exit if game is over
        if self.state.done():
            return self

        # If child node didn't exist we need to expand root 
        if child is None:
            self.root = self.expand_root()

        return self

    def expand_root (self):
        logits, value = self.model.infer({'image' : [self.state.observation()]})
        self.root_value = value[0][0]
        return self.expand(self.state.actions(), logits[0])

    def expand (self, actions, logits):
        # Legal actions in this state
        actions = sorted (
            actions,
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

        return node

    def print_tree (self):
        graph = []
        graph.append('strict digraph {')

        self.print_nodes(self.root, self.root_value, graph)
        self.print_edges(self.root, graph)

        graph.append('}')
        return '\n'.join(graph)

    def print_nodes (self, node, value, graph):
        key = id(node)

        graph.append('\t"{}" [label={:.2f}]'.format(key, value))

        for index in node.children:
            child = node.children[index]
            action = node.actions[index]

            if node.N[index]:
                self.state.push_action(action)
                self.print_nodes(child, node.Q[index], graph)
                self.state.pop_action()

        return graph

    def print_edges (self, node, graph):
        key = id(node)

        for index in node.children:
            child = node.children[index]
            child_key = id(child)

            action = node.actions[index]
            move = self.state.parse_action(action)

            if node.N[index]:
                graph.append (
                    '\t"{}" -> "{}" [label="{}, {}"]'.format (
                        key, child_key, self.state.state.san(move), node.N[index]
                    )
                )

                self.state.push_action(action)
                self.print_edges(child, graph)
                self.state.pop_action()
