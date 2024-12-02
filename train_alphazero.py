'''
main alphazero training file
takes in all necessary components and defines the overall alphazero training loop
'''

from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dots_boxes.game_logic import *
from dots_boxes.nnet import *


class AZLoss(nn.Module):
    def __init__(self, wd):
        super(AZLoss, self).__init__()

    """
    "Specifically, the parameters θ are adjusted by gradient 
    descent on a loss function l that sums over the mean-squared error and 
    cross-entropy losses, respectively:
    (p,v)=f_θ(s) and l=(z-v)^2 - π^Tlog(p)    (1)" (AlphaGo Zero 355)
    note that weight decay can be handled in the optimizer

    For some reason, they don't have a hyperparameter between loss terms. Not sure why.
    """
    def forward(self, pi, p, z, v):
        v_err = (z - v)**2 # mean squared error
        p_err = torch.sum(pi*p, dim=-1)
        loss = (v_err.view(-1) - p_err).mean()
        # missing the weight regularization, maybe not needed for t3
        return loss

class Trainer(object):
    def __init__(self, hparams):
        '''
        Use (nested) easydict for hyperparameters!
        hparams can include:
        - num iterations per training round
        - number of epochs
        - optimizer hparams
        - convergence threshold
        - dataloader config
        - etc.
        '''
        pass

    def train_model(self, net, dataset):
        '''
        Define typical training loop given a network and dataset.
        Return the trained model
        '''
        return None


class AlphaZero(object):
    def __init__(self, config):
        '''
        Use easydict to create hyperparameter config! (JSON-style dict)
        '''
        pass

    def run_training(self):
        '''
        main training function
        i believe in you lol
        you can follow from https://arxiv.org/pdf/1903.08129 page 5
        '''
        pass

    def self_play_game(self, p1, p2=None, eval=False):
        '''
        simulates a game between 2 players (aka networks) p1 and p2
        eval=False -> tracks and updates storage with the relevant states/policies/final values
        eval=True -> play greedy moves to compare the two networks
        '''
        if p2 is None:
            p2 = p1

    def compare_models(self, new_model):
        '''
        compare 2 models and return the best one
        play k games between them and update to new_model if high enough winrate
        '''
        return None


def main():
    '''
    Main function! Train the alphazero instance, save the best model, all that good stuff.
    '''
    pass

if __name__ == '__main__':
    main()