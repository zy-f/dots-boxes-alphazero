'''
defines a Storage class for dataset creation/update as the model undergoes self-play,
as well as comparing versions of the model to each other to determine which to keep
'''

from torch.utils.data import Dataset
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from dots_boxes.game_logic import *

class DnBStorageDataset(Dataset):
    def __init__(self, states, policies, values):
        super(DnBStorageDataset, self).__init__()
        boards, scores = zip(*states)
        self.board_states = torch.from_numpy(np.stack(boards))
        self.score_states = torch.tensor(scores, dtype=torch.float32)
        self.policies = torch.from_numpy(np.stack(policies))
        self.values = torch.from_numpy(np.stack(values))

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return (
            (self.board_states[idx], self.score_states[idx]),
            self.policies[idx],
            self.values[idx],
        )
    
class Storage(object):
    def __init__(self, config, board_size):
        '''
        Initialize the storage object with relevant stuff idk
        Needs to track the running buffer of data and the current best network
        config inputs:
        - buffer_size: max number of data points to store
        - ckpt_dir: directory to save networks to
        - exp_name: experiment name (sets the save filename for the network)
        '''
        self.cfg = config
        self.board = DnBBoard(num_boxes=board_size)
        self.buffer = {"states": [], "policies": [], "values": []}
        self.best_net = None
        self.folder = f"{self.cfg.ckpt_dir}/{self.cfg.exp_name}"
        os.makedirs(self.folder, exist_ok=True)

        # for saving winrate against baseline files
        file_names = ["winrate_against_greedy.txt", "winrate_against_random.txt", "train_loss.txt"]
        for file_name in file_names:
            file_path = os.path.join(self.folder, file_name)
            with open(file_path, 'w') as file:
                pass
        
        augmentation = self.cfg.get("augmentation", False)
        print(f"\nUsing augmentation: {augmentation}")
    
    def update_winrate(self, winrate, baseline):
        with open(os.path.join(self.folder, f"winrate_against_{baseline}.txt"), 'a') as file:
            file.write(f"{winrate}\n")
    
    def update_train_loss(self, train_loss):
        with open(os.path.join(self.folder, f"train_loss.txt"), 'a') as file:
            file.write(f"{train_loss}\n")
    
    def plot_winrates(self):
        with open(os.path.join(self.folder, "winrate_against_greedy.txt"), 'r') as file:
            greedy_winrates = [float(line.strip()) for line in file.readlines()]
        
        with open(os.path.join(self.folder, "winrate_against_random.txt"), 'r') as file:
            random_winrates = [float(line.strip()) for line in file.readlines()]
        
        iterations = range(1, len(greedy_winrates) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, greedy_winrates, label='Win Rate Against Greedy', color='b', linewidth=2)
        plt.plot(iterations, random_winrates, label='Win Rate Against Random', color='g', linewidth=2)
        plt.axhline(y=0.5, color='gray', linestyle='--')

        plt.ylim(0, 1)
        plt.title('Winrate Evaluation Curves', fontsize=16)
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Winrate', fontsize=14)
        
        plt.legend()
        plt.savefig(os.path.join(self.folder, "winrate_plots.png"), dpi=300)
        plt.close()
    
    def plot_train_loss(self):
        with open(os.path.join(self.folder, "train_loss.txt"), 'r') as file:
            train_loss = [float(line.strip()) for line in file.readlines()]
        
        iterations = range(1, len(train_loss) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, train_loss, label='Train loss', color='r', linewidth=2)

        plt.title('Training Loss Curve', fontsize=16)
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Training Loss', fontsize=14)
        
        plt.legend()
        plt.savefig(os.path.join(self.folder, "train_loss.png"), dpi=300)
        plt.close()

    def best_network(self):
        '''
        returns the best network currently tracked
        '''
        return self.best_net
    
    def save_network(self, net):
        '''
        saves a new network as the best network going forward
        '''
        self.best_net = net
        ckpt_path = f"{self.folder}/{self.cfg.exp_name}.pth"
        torch.save(net.state_dict(), ckpt_path)
        print(f"Best network saved to {ckpt_path}")
    
    def get_dataset(self):
        '''
        returns a torch Dataset of the current data buffer with policy and value labels
        '''

        states = self.buffer["states"]
        policies = self.buffer["policies"]
        values = self.buffer["values"]

        return DnBStorageDataset(states, policies, values)
    
    def add_data(self, states, policies, values):
        '''
        states, policies, and values, are parallel lists or arrays
        load these into an appropriate structure, managing the data buffer as necessary
        '''
        augmentation = self.cfg.get("augmentation", False)
        if augmentation:
            new_states, new_policies, new_values = self.add_symmetrical_data(states, policies, values)
            self.buffer["states"].extend(new_states)
            self.buffer["policies"].extend(new_policies)
            self.buffer["values"].extend(new_values)
        else:
            self.buffer["states"].extend(states)
            self.buffer["policies"].extend(policies)
            self.buffer["values"].extend(values)

        if self.cfg.buffer_size > 0:
            current_size = len(self.buffer["states"])
            if current_size > self.cfg.buffer_size:
                overflow = current_size - self.cfg.buffer_size
                self.buffer["states"] = self.buffer["states"][overflow:]
                self.buffer["policies"] = self.buffer["policies"][overflow:]
                self.buffer["values"] = self.buffer["values"][overflow:]
    
    def add_symmetrical_data(self, states, policies, values):
        new_states = []
        new_policies = []
        new_values = []

        for state, policy, value in zip(states, policies, values):
            # collect all transformations
            transformations = [
                (state, policy),  # Original
                self.board.rotate_90(state, policy),
                self.board.rotate_180(state, policy),
                self.board.rotate_270(state, policy),
                self.board.reflect_x(state, policy),
                self.board.reflect_y(state, policy),
                self.board.reflect_y(*self.board.rotate_90(state, policy)),  # 90° rotation + reflect y
                self.board.reflect_x(*self.board.rotate_270(state, policy)),  # 270° rotation + reflect x
            ]

            for transformed_state, transformed_policy in transformations:
                new_states.append(transformed_state)
                new_policies.append(transformed_policy)
                new_values.append(value)

        return new_states, new_policies, new_values


def debug():
    # TODO: probably should change this!

    storage = Storage(buffer_size=1000)
    states = [torch.randn(4, 3, 3) for _ in range(10)]
    policies = [torch.rand(10) for _ in range(10)]
    values = [torch.tensor(0.5) for _ in range(10)]

    storage.add_data(states, policies, values)
    dataset = storage.get_dataset()
    print(dataset[0])

if __name__ == "__main__":
    debug()