'''
main alphazero training file
takes in all necessary components and defines the overall alphazero training loop
'''

import sys
import yaml
import traceback
import time
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np
from dots_boxes.game_logic import *
from dots_boxes.nnet import *
from data_store import *
from mcts import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from dots_boxes.play_logic import *

class AZLoss(nn.Module):
    def __init__(self):
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
            collate_fn = type(dataset).collate_fn \
                if hasattr(dataset, 'collate_fn') else None,
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
        pbar = trange(self.hparams.epochs)
        for epoch in pbar:

            # TODO: haven't handled any convergence criterion yet

            epoch_loss = 0.0
            batch_count = 0
            for (states, pi, z) in dataloader:
                p, v = net(states)

                loss = self.loss_fn(pi, p, z, v)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
            pbar.set_description(f"Epoch {epoch + 1}/{self.hparams.epochs}, Loss: {epoch_loss / batch_count:.4f}")
        
        return net.cpu()


class AlphaZero(object):
    def __init__(self, config):
        '''
        Use easydict to create hyperparameter config! (JSON-style dict)
        '''
        self.config = config
        self.storage = Storage(config.storage_config)
        self.board = DnBBoard(num_boxes=config.num_boxes)
        self.current_net = DnBNet(self.board.nb, len(self.board.action_mapping))
        self.storage.save_network(self.current_net)
        self.trainer = Trainer(config.trainer_hparams)
        self.mcts = MCTS(config.mcts_config)
        self.rng = np.random.default_rng(seed=config.seed)

        self.greedy_player = GreedyBaselineBot()
        self.random_player = RandomBaselineBot()

    def run_training(self):
        '''
        main training function
        i believe in you lol
        you can follow from https://arxiv.org/pdf/1903.08129 page 5
        '''
        for iteration in range(self.config.alphazero_iterations):
            print(f"=== Iteration {iteration + 1}/{self.config.alphazero_iterations} ===")

            # 1. self-play
            # start_time = time.time()
            self.parallel_self_play(self.config.self_play_games_per_iter, self.current_net)
            # print(f"Training self play takes {time.time() - start_time:.3f} seconds.")

            # 2. train
            dataset = self.storage.get_dataset()
            trained_net = self.trainer.train_model(self.current_net, dataset)

            # 3. compare models
            if self.compare_models(trained_net):
                self.storage.save_network(trained_net)
                self.current_net = trained_net
            
            winrate_against_greedy = self.compare_models(trained_net, baseline = "greedy")
            self.storage.update_winrate(winrate_against_greedy, baseline = "greedy")
            winrate_against_random = self.compare_models(trained_net, baseline = "random")
            self.storage.update_winrate(winrate_against_random, baseline = "random")
            self.storage.plot_winrates()
            
            print()

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

        n_moves = 0
        while board.end_value() is None:
            net = p1 if board.player_turn == 0 else p2

            if isinstance(net, nn.Module):
                pi = self.mcts.search(board, net)
                if eval or n_moves > self.config.optimal_move_cutoff:
                    move = np.argmax(pi)  # play greedily
                else:
                    move = self.rng.choice(np.arange(len(pi)), p=pi)  # sample a move

                if not eval:
                    if board.player_turn == 0: 
                        states1.append(board.nn_state())
                        policies1.append(pi)
                    else:
                        states2.append(board.nn_state())
                        policies2.append(pi)
            else:
                # baselines
                assert(eval)
                move = net.move(board)

            board.play(move)
            n_moves += 1

        winner = board.end_value()  # +1 for player 1, -1 for player 2, 0 for draw

        if not eval:
            values1 = [winner for _ in states1]
            # self.storage.add_data(states1, policies1, values1)
            values2 = [-winner for _ in states2]
            # self.storage.add_data(states2, policies2, values2)
            return winner, states1, policies1, values1, states2, policies2, values2
        else:
            return winner

    def parallel_self_play(self, num_games, p1, p2=None, eval=False):
        num_workers = min(8, num_games)
        if not eval:
            tasks = [(p1, p2, eval) for _ in range(num_games)]
        else:
            # switch position
            tasks = [(p1, p2, eval) if i%2 == 0 else (p2, p1, eval) for i in range(num_games)]
        results = [None] * num_games

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.self_play_game, *task): i for i, task in enumerate(tasks)}
            for future in tqdm(as_completed(futures), total=num_games, desc="Self-Play Games"):
                try:
                    index = futures[future]
                    if not eval:
                        winner, states1, policies1, values1, states2, policies2, values2 = future.result()
                        self.storage.add_data(states1, policies1, values1)
                        self.storage.add_data(states2, policies2, values2)
                    else:
                        winner = future.result()
                    results[index] = winner
                except Exception as e:
                    print(f"Error during self-play: {e}")
                    print(traceback.print_exc())
        return results

    def compare_models(self, new_model, baseline = None):
        '''
        compare 2 models and return the best one
        play k games between them and update to new_model if high enough winrate
        '''
        if baseline is None:
            current_best = self.storage.best_network()  # prev best model
        elif baseline == "greedy":
            current_best = self.greedy_player
        elif baseline == "random":
            current_best = self.random_player

        new_model_wins = 0
        total_games = self.config.comparison_games_per_iter
        
        # start_time = time.time()
        winners = self.parallel_self_play(total_games, new_model, current_best, eval=True)
        # print(f"Comparison self play takes {time.time() - start_time:.3f} seconds.")

        for i, winner in enumerate(winners):
            if winner == 0:
                new_model_wins += 0.5  # TODO: draw counts as 0.5?
            if i % 2 == 0:
                if winner == +1:
                    new_model_wins += 1
            else:
                if winner == -1:
                    new_model_wins += 1

        win_rate = new_model_wins / total_games
        if baseline is None:
            print('New model winrate: ', f"{win_rate:.3f}")
        elif baseline == "greedy":
            print('New model against greedy winrate: ', f"{win_rate:.3f}")
            return win_rate
        elif baseline == "random":
            print('New model against random winrate: ', f"{win_rate:.3f}")
            return win_rate

        return win_rate > self.config.comparison_update_thresh

def main(config_path):
    '''
    Main function! Train the alphazero instance, save the best model, all that good stuff.
    '''
    # TODO: some way to load the config
    with open(config_path, 'r') as f:
        config = edict(yaml.safe_load(f))
    alpha_zero = AlphaZero(config)
    alpha_zero.run_training()

if __name__ == '__main__':
    assert len(sys.argv) > 1
    config_path = sys.argv[1]
    main(config_path)