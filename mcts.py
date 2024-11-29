from easydict import EasyDict as edict
from tqdm import trange
from collections import defaultdict
import numpy as np

EPS = 1e-8

'''
implements the Monte-Carlo Tree Search step of the algorithm
i'm doing this i guess
'''

def dirichlet_noise(noise, size):
    return np.random.gamma(noise, scale=1, size=size)

class MCSearchTree(object):
    def __init__(self, board, net, params=None):
        '''
        adapted from https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py
        '''
        self.root_board = board
        self.net = net
        self.Qsa = defaultdict(float)
        self.Nsa = defaultdict(int)
        self.Ns = {}
        self.state_policies = {} # stores (policy, legal_move_mask) tuples
        self.params = params
        self.n_actions = len(board.legal_action_mask())
        self.noise_distrib = dirichlet_noise(params.noise, size=self.n_actions)

    def rollout_to_leaf(self, board=None):
        '''
        recursively performs rollout to leaf; returns -z (z = board value)
        (negated since value is flipped for the other player)
        use board=None to indicate that we are at the root node
        '''
        is_root = board is None
        if is_root:
            board = self.root_board.clone()
        ## stop cases
        if z := board.end_value() is not None: # stop case 1: game ends
            # z is player-1 based, should only return negative if the previous player is 1
            return z if board.player_turn else -z
        ts = board.tree_state()
        # stop case 2: leaf node reached
        if ts not in self.Ns:
            self.Ns[ts] = 0
            nn_p, nn_z = self.net.predict(board.nn_state())
            if is_root: # add noise only for root
                frac = self.params.noise_frac
                nn_p = (1-frac)*nn_p + frac*self.noise_distrib
            legal_mask = board.legal_action_mask()
            if (nn_p == 0).all(): # edge case: bad nn output
                nn_p = np.ones_like(nn_p)
            p = nn_p * legal_mask
            p /= p.sum() # renormalize
            self.state_policies[ts] = (p, legal_mask)
            # leaf found, backup the rollout
            # return negative always since nn returns score for current player
            return -z
        ## if not in a stop case, need to pick an action!
        self.Ns[ts] += 1 # doing this cuz it makes more sense to me but not sure it's right
        p, legal_mask = self.state_policies[ts]
        U = np.array([
            self.Qsa[(ts, a)] + self.params.c_puct * p[a] * np.sqrt(self.Ns[ts]) \
                / (1 + self.Nsa[(ts, a)])
            for a in np.where(legal_mask)[0] # iterate thru legal actions
        ])
        a_max = np.argmax(U)
        was_legal = board.play(a_max)
        if not was_legal:
            raise ValueError('!? move selected was illegal')
        z = self.rollout_to_leaf(board=board)
        s_a = (ts, a_max)
        # iterative Qsa and Nsa updates, reduces to Q=z and N=1 in the new case
        self.Qsa[s_a] = (self.Nsa[s_a] * self.Qsa[s_a] + z) / (self.Nsa[s_a] + 1)
        self.Nsa[s_a] += 1
        return -z # continue the backup process

    def root_counts(self):
        s = self.root_board.tree_state()
        return np.array([self.Nsa[(s,a)] for a in range(len(self.n_actions))])


class MCTS(object):
    
    def __init__(self, verbose=False, params=edict(
        n_sim=100,
        c_puct=1,
        noise=.03,
        noise_frac=0.25,
        tau_pi=1
    )):
        '''
        search parameters:
            - n_sim = number of rollouts to perform before selecting the next move
            - c_puct = exploration vs exploitation parameter
            - noise = dirichlet noise added to move probabilities in root node
            - noise_frac = weight of noise vs true probabilitie for the root 
            - tau_pi = standard temperature param for softening probabilities in pi
        '''
        self.params = params
        self.verbose = verbose
    
    def search(self, board, net, tau=None):
        '''
        inputs:
            - board = the starting board state (class instance, not just the representation)
            - net = the neural network that returns policies and values
            - tau = tau_pi from the initializer, but with an override option
        returns: move probabilities pi across all actions for the next move
        '''
        if tau is None:
            tau = self.params.tau_pi
        tree = MCSearchTree(board, net, params=self.params)
        for _ in range(self.params.n_sim):
            tree.rollout_to_leaf()
        Nsa = tree.root_counts()
        if tau == 0:
            pi = np.zeros(Nsa.shape)
            pi[np.argmax(Nsa)] = 1
        else:
            pi = Nsa**(1/tau)
            pi /= np.sum(pi)
        return pi