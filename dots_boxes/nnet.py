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

    def __init__(self, board_size, action_space, num_filters=64, num_res_blocks=3):
        super(DnBNet, self).__init__()

        print(f"\nUsing {num_filters} filter size and {num_res_blocks} res blocks!.\n")
        self.board_size = board_size
        self.max_score = board_size * board_size  # used to normalize score to 0 to 1 
        self.action_space = action_space

        self.conv1 = nn.Conv2d(4, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        self.global_conv = nn.Conv2d(num_filters, num_filters, kernel_size=1)
        self.global_bn = nn.BatchNorm2d(num_filters)
        self.global_fc = nn.Linear(num_filters * board_size * board_size + 2, num_filters)

        self.policy_fc = nn.Linear(num_filters, action_space)
        self.value_fc = nn.Linear(num_filters, 1)

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
        s = s / self.max_score

        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.bn1(self.conv1(x)))

        for block in self.res_blocks:
            x = block(x)

        x = F.relu(self.global_bn(self.global_conv(x)))
        x = x.contiguous().view(batch_size, -1)
        x = torch.cat((x, s), dim=1)  # incorporate score information
        x = F.relu(self.global_fc(x))

        # policy
        p = F.log_softmax(self.policy_fc(x), dim=1)

        # value
        v = torch.tanh(self.value_fc(x)).squeeze()

        return p, v

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


class DnBNetFC(nn.Module):
    def __init__(self, board_size, action_space, dim=256, dropout=0.3):
        super(DnBNetFC, self).__init__()

        print(f"\nUsing a fully connected architecture with hidden dimension={dim} and dropout={dropout}.\n")
        self.board_size = board_size
        self.action_space = action_space

        input_size = board_size * board_size * 4 + 2
        self.fc1 = nn.Linear(input_size, dim)
        self.bn1 = nn.BatchNorm1d(dim)

        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

        self.fc3 = nn.Linear(dim, dim)
        self.bn3 = nn.BatchNorm1d(dim)

        self.fc4 = nn.Linear(dim, dim)
        self.bn4 = nn.BatchNorm1d(dim)

        self.fc5 = nn.Linear(dim, dim)
        self.bn5 = nn.BatchNorm1d(dim)

        self.fc6 = nn.Linear(dim, dim)
        self.bn6 = nn.BatchNorm1d(dim)

        self.policy_fc = nn.Linear(dim, action_space)
        self.value_fc = nn.Linear(dim, 1)

        self.dropout = nn.Dropout(dropout)

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
        s = s / (self.board_size * self.board_size)  # Normalize scores

        batch_size = x.size(0)
        x = x.contiguous().view(batch_size, -1)  # Flatten the board state
        x = torch.cat((x, s), dim=1)  # Combine board state and scores

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)

        x = F.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)

        x = F.relu(self.bn6(self.fc6(x)))
        x = self.dropout(x)

        # policy
        p = F.log_softmax(self.policy_fc(x), dim=1)

        # value
        v = torch.tanh(self.value_fc(x)).squeeze()

        return p, v
    
    def predict(self, board_state):
        """
        Parameters:
        - board_state: tuple, a single (board state, scores) pair.

        Returns:
        - p: np.ndarray, policy vector of probabilities for the action space.
        - z: np.ndarray, scalar state evaluation.
        """
        board, score = board_state
        board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
        score_tensor = torch.tensor(score, dtype=torch.float32).unsqueeze(0)
        self.eval()
        with torch.no_grad():
            p, z = self.forward((board_tensor, score_tensor))
            p = torch.exp(p)
        
        return p.squeeze(0).cpu().numpy(), z.item()

if __name__ == "__main__":
    debug()