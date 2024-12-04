'''
main alphazero training file
takes in all necessary components and defines the overall alphazero training loop
'''

from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from dots_boxes.game_logic import *
from dots_boxes.nnet import *
from data_store import *
from mcts import *


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
        - number of epochs
        - optimizer hparams
        - convergence threshold
        - dataloader config
        - etc.
        '''

        # TODO: assumes hparams contains batch_size, num_workers, lr, weight_decay, epochs
        self.hparams = hparams
        self.loss_fn = AZLoss()

    def train_model(self, net, dataset):
        '''
        Define typical training loop given a network and dataset.
        Return the trained model
        '''

        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.get("num_workers", 4),
        )

        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.get("weight_decay", 0),
        )

        net.train()
        for epoch in range(self.hparams.epochs):

            # TODO: haven't handled any convergence criterion yet

            epoch_loss = 0.0
            for batch in dataloader:
                states = batch["state"]
                pi = batch["policy"]
                z = batch["value"]

                p, v = net(states)

                loss = self.loss_fn(pi, p, z, v)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{self.hparams.epochs}, Loss: {epoch_loss:.4f}")
        
        return net



class AlphaZero(object):
    def __init__(self, config):
        '''
        Use easydict to create hyperparameter config! (JSON-style dict)
        '''
        self.config = config
        self.config = config
        self.storage = Storage(config.buffer_size)
        self.board = DnBBoard(num_boxes=config.num_boxes)
        self.current_net = DnBNet(self.board.nb, len(self.board.action_mapping))
        self.storage.save_network(self.current_net)
        self.trainer = Trainer(config.trainer_hparams)
        self.mcts = MCTS()

    def run_training(self):
        '''
        main training function
        i believe in you lol
        you can follow from https://arxiv.org/pdf/1903.08129 page 5
        '''
        for iteration in range(self.config.num_iterations):
            print(f"=== Iteration {iteration + 1}/{self.config.num_iterations} ===")

            # 1. self-play
            for _ in range(self.config.num_self_play_games):
                self.self_play_game(self.current_net)

            # 2. train
            dataset = self.storage.get_dataset()
            trained_net = self.trainer.train_model(self.current_net, dataset)

            # 3. compare models
            if self.compare_models(trained_net):
                self.storage.save_network(trained_net)
                self.current_net = trained_net
            else:
                # TODO: do some early stopping stuff?
                pass

    def self_play_game(self, p1, p2=None, eval=False):
        '''
        simulates a game between 2 players (aka networks) p1 and p2
        eval=False -> tracks and updates storage with the relevant states/policies/final values
        eval=True -> play greedy moves to compare the two networks
        '''
        if p2 is None:
            p2 = p1
        
        board = DnBBoard(num_boxes=self.config.num_boxes)
        states1, policies1, values1 = [], [], []
        states2, policies2, values2 = [], [], []

        while not board.end_value() is None:
            net = p1 if board.player_turn == 0 else p2
            pi = self.mcts.search(board, net)

            if eval:
                move = torch.argmax(pi).item()  # play greedily
            else:
                move = torch.multinomial(pi, 1).item()  # sample a move

            if not eval:
                if board.player_turn == 0: 
                    states1.append(board.nn_state())
                    policies1.append(pi.cpu().numpy())
                else:
                    states2.append(board.nn_state())
                    policies2.append(pi.cpu().numpy())

            board.play(move)

        winner = board.end_value()  # +1 for player 1, -1 for player 2, 0 for draw

        if not eval:
            values1 = [winner for _ in states1]
            self.storage.add_data(states1, policies1, values1)
            values2 = [-winner for _ in states2]
            self.storage.add_data(states2, policies2, values2)

        return winner

    def compare_models(self, new_model):
        '''
        compare 2 models and return the best one
        play k games between them and update to new_model if high enough winrate
        '''
        current_best = self.storage.best_network()
        new_model_wins = 0
        total_games = self.config.num_comparison_games

        for game in range(total_games):
            # alternate which model plays as p1
            if game % 2 == 0:
                winner = self.self_play_game(new_model, current_best, eval=True)
                if winner == +1:
                    new_model_wins += 1
            else:
                winner = self.self_play_game(current_best, new_model, eval=True)
                if winner == -1:
                    new_model_wins += 1

        win_rate = new_model_wins / total_games
        return win_rate >= 0.5 # if wins more than 50% of time?

def main():
    '''
    Main function! Train the alphazero instance, save the best model, all that good stuff.
    '''
    # TODO: some way to load the config
    config = None
    alpha_zero = AlphaZero(config)
    alpha_zero.run_training()

if __name__ == '__main__':
    main()