'''
defines a Storage class for dataset creation/update as the model undergoes self-play,
as well as comparing versions of the model to each other to determine which to keep
'''

from torch.utils.data import Dataset

class Storage(object):
    def __init__(self, buffer_size=-1):
        '''
        Initialize the storage object with relevant stuff idk
        Needs to track the running buffer of data and the current best network
        '''
        pass

    def best_network(self):
        '''
        returns the best network currently tracked
        '''
        return None
    
    def save_network(self, net):
        '''
        saves a new network as the best network going forward
        '''
        pass
    
    def get_dataset(self):
        '''
        returns a torch Dataset of the current data buffer with policy and value labels
        '''
        return None
    
    def add_data(self, states, policies, values):
        '''
        states, policies, and values, are parallel lists or arrays
        load these into an appropriate structure, managing the data buffer as necessary
        '''
        pass