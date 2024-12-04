'''
defines a Storage class for dataset creation/update as the model undergoes self-play,
as well as comparing versions of the model to each other to determine which to keep
'''

from torch.utils.data import Dataset
import torch
import numpy as np
import os

class StorageDataset(Dataset):
    def __init__(self, states, policies, values):
        super(StorageDataset, self).__init__()
        self.states = torch.from_numpy(states)
        self.policies = torch.from_numpy(policies)
        self.values = torch.from_numpy(values)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "state": self.states[idx],
            "policy": self.policies[idx],
            "value": self.values[idx],
        }
    
class Storage(object):
    def __init__(self, config):
        '''
        Initialize the storage object with relevant stuff idk
        Needs to track the running buffer of data and the current best network
        config inputs:
        - buffer_size: max number of data points to store
        - ckpt_dir: directory to save networks to
        - exp_name: experiment name (sets the save filename for the network)
        '''
        self.cfg = config
        self.buffer = {"states": [], "policies": [], "values": []}
        self.best_net = None
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)

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
        ckpt_path = f"{self.cfg.ckpt_dir}/{self.cfg.exp_name}.pth"
        torch.save(net.state_dict(), ckpt_path)
        print(f"Best network saved to {ckpt_path}")
    
    def get_dataset(self):
        '''
        returns a torch Dataset of the current data buffer with policy and value labels
        '''

        states = np.stack(self.buffer["states"])
        policies = np.stack(self.buffer["policies"])
        values = np.stack(self.buffer["values"])

        return StorageDataset(states, policies, values)
    
    def add_data(self, states, policies, values):
        '''
        states, policies, and values, are parallel lists or arrays
        load these into an appropriate structure, managing the data buffer as necessary
        '''

        # TODO: do we only get one value per game?
        # TODO: would the inputs be tensors or arrays?
        # TODO: don't we also need the predicted action probabilities and the predicted value?

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