'''
defines a Storage class for dataset creation/update as the model undergoes self-play,
as well as comparing versions of the model to each other to determine which to keep
'''

from torch.utils.data import Dataset
import torch
import os

class StorageDataset(Dataset):
    def __init__(self, states, policies, values):
        super(StorageDataset, self).__init__()
        self.states = states
        self.policies = policies
        self.values = values

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "state": self.states[idx],
            "policy": self.policies[idx],
            "value": self.values[idx],
        }
    
class Storage(object):
    def __init__(self, buffer_size=-1):
        '''
        Initialize the storage object with relevant stuff idk
        Needs to track the running buffer of data and the current best network
        '''
        self.buffer_size = buffer_size # this is the maximum number of data points to store?
        self.buffer = {"states": [], "policies": [], "values": []}
        self.best_net = None
        self.ckpt_dir = "./ckpt"
        os.makedirs(self.ckpt_dir, exist_ok=True)

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
        ckpt_path = os.path.join(self.ckpt_dir, "best_network.pth")
        torch.save(net.state_dict(), ckpt_path)
        print(f"Best network saved to {ckpt_path}")
    
    def get_dataset(self):
        '''
        returns a torch Dataset of the current data buffer with policy and value labels
        '''

        states = torch.stack(self.buffer["states"])
        policies = torch.stack(self.buffer["policies"])
        values = torch.stack(self.buffer["values"])

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

        if self.buffer_size > 0:
            current_size = len(self.buffer["states"])
            if current_size > self.buffer_size:
                overflow = current_size - self.buffer_size
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