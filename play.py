'''
file to allow interaction between user and a trained alphazero instance
basically the inference-time file that lets you play vs the bot user
'''
import dots_boxes
from mcts import MCTS
import numpy as np
from dots_boxes.nnet import *
import sys
import yaml
from easydict import EasyDict as edict

GAMES = {
    'dnb': dots_boxes
}

class Player:
    is_human = False
    label = "null"
    def __init__(self, *args):
        pass
    def move(self, board):
        pass

class AlphaZero(Player):
    label = "AlphaZero (bot)"
    def __init__(self, num_boxes, game='dnb'):
        self.board = DnBBoard(num_boxes=num_boxes)
        root_path = f"final_models/size{num_boxes}_aug_final"
        with open(root_path+".yaml", 'r') as f:
            config = edict(yaml.safe_load(f))
        self.net = DnBNet(self.board.nb, len(self.board.action_mapping),
                          num_filters=config.model_config.num_filters, 
                          num_res_blocks=config.model_config.num_res_blocks)
        
        self.mcts = MCTS(config.mcts_config)
        self.pretrained_path = root_path+".pth"
        checkpoint = torch.load(self.pretrained_path, map_location=torch.device('cpu'))
        self.net.load_state_dict(checkpoint)

    def move(self, board):
        policy = self.mcts.search(board, self.net)
        return np.argmax(policy)


def play(game='dnb'):
    '''
    We'll probably want either a regular or command-line argument to choose 
    the trained network to play against.
    '''
    player_options = GAMES[game].PLAYER_TYPES
    player_options['alphazero'] = AlphaZero
    show_computer_moves = input('Show computer moves (Y/n)? ')
    verbose = show_computer_moves.lower().startswith('y')
    game = GAMES[game].Game()
    while not game.finished():
        if verbose or game.human_turn():
            print()
            game.display()
        game.play_turn()
    print("\n\n--- Game End ---")
    game.display()


if __name__ == '__main__':
    play()