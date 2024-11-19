'''
file for the neural net for our alphazero implementation. defines:
- converting game_logic types of states into neural net input state
- network arch to map neural net input state -> policy and value head outputs
'''

class DnBNet(nn.Module):
    '''
    Defines a model architecture for our AlphaZero policy/value predictive network. 
    - init: define appropriate parameters given board size and anything else we want to tune
    - forward: given a batch of board states OR single board state, return
        - policy vector p containing probabilities in the action space (legal moves handled in mcts)
        - scalar state evaluation z (+1 if 100% winning, -1 if 100% losing -> use a tanh)
    '''
    pass