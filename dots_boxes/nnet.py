'''
file for the neural net for our alphazero implementation. defines:
- converting game_logic types of states into neural net input state
- network arch to map neural net input state -> policy and value head outputs
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dots_boxes.game_logic import *

class DnBNet(nn.Module):
    '''
    Defines a model architecture for our AlphaZero policy/value predictive network. 
    - init: define appropriate parameters given board size and anything else we want to tune
    - forward: given a batch of board states, return
        - policy vector p containing probabilities in the action space (legal moves handled in mcts)
        - scalar state evaluation z (+1 if 100% winning, -1 if 100% losing -> use a tanh)
    - predict: given a single (board state, scores) tuple (elements NOT given as tensors), return
        - policy vector p containing probabilities in the action space (legal moves handled in mcts)
        - scalar state evaluation z (+1 if 100% winning, -1 if 100% losing -> use a tanh)
        - NOTE: values should be returned as numpy arrays, not tensors
    '''

    def __init__(self, board_size, action_space, num_filters=32, num_res_blocks=3):
        super(DnBNet, self).__init__()
        self.board_size = board_size
        self.action_space = action_space

        self.conv1 = nn.Conv2d(4, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        self.policy_conv = nn.Conv2d(num_filters, 4, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * board_size * board_size, action_space)

        self.value_conv = nn.Conv2d(num_filters, 4, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc = nn.Linear(4 * board_size * board_size + 2, 1)

    def forward(self, state):
        """
        Parameters:
        state is a tuple of:
        - x: torch.Tensor, batch of board states with shape (batch_size, board_size, board_size, 4).
        - s: torch.Tensor, batch of scores (i.e. [my score, their score]) with shape (batch_size, 2).

        Returns:
        - p: torch.Tensor, policy vector of shape (batch_size, action_space).
        - z: torch.Tensor, scalar state evaluation of shape (batch_size, 1).
        """
        x, s = state
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.bn1(self.conv1(x)))

        for block in self.res_blocks:
            x = block(x)

        # policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.contiguous().view(batch_size, -1)
        p = F.log_softmax(self.policy_fc(p), dim=1)

        # value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.contiguous().view(batch_size, -1)
        v = torch.cat((v, s), dim=1)
        z = torch.tanh(self.value_fc(v)).squeeze()

        return p, z

    def predict(self, board_state):
        """
        Parameters:
        - board_state: tuple, a single (board state, scores) pair.

        Returns:
        - p: np.ndarray, policy vector of probabilities for the action space.
        - z: np.ndarray, scalar state evaluation.
        """

        # TODO: how are we handling the score? Should it also be an input?

        board, score = board_state
        board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
        score_tensor = torch.tensor(score, dtype=torch.float32).unsqueeze(0)
        self.eval()
        with torch.no_grad():
            p, z = self.forward((board_tensor, score_tensor))
            p = torch.exp(p)
        
        return p.squeeze(0).cpu().numpy(), z.item()


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)
    

def debug():
    nb = 2
    board = DnBBoard(num_boxes=nb)
    model = DnBNet(board.nb, len(board.action_mapping))
    print(board)
    while 1:
        p, z = model.predict(board.nn_state())
        print(p, z)
        inp = input(f"player {board.player_turn+1}: ").strip()
        if inp.lower() == 'q':
            break
        try:
            move = int(inp)
        except:
            move = inp.split()
        board.play(move)
        print(board)
        z = board.end_value()
        if z is not None:
            print(z)
            break


if __name__ == "__main__":
    debug()