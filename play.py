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
    def move(self, board):
        pass

class AlphaZero(Player):
    label = "AlphaZero (bot)"
    def __init__(self, config, game='dnb'):
        # self.net = GAMES[game].Network(pretrained=True)
        self.config = config
        self.board = DnBBoard(num_boxes=config.num_boxes)
        
        if self.config.model_config.get("fc", False):
            self.net = DnBNetFC(self.board.nb, len(self.board.action_mapping))
        else:
            self.net = DnBNet(self.board.nb, len(self.board.action_mapping),
                                    num_filters=config.model_config.num_filters, 
                                    num_res_blocks=config.model_config.num_res_blocks)
        
        self.mcts = MCTS(config.mcts_config)
        self.pretrained_path = f"{self.config.storage_config.ckpt_dir}/{self.config.storage_config.exp_name}/{self.config.storage_config.exp_name}.pth"
        checkpoint = torch.load(self.pretrained_path, map_location=torch.device('cpu'))
        self.net.load_state_dict(checkpoint)

    def move(self, board):
        policy = self.mcts.search(board, self.net)
        return np.argmax(policy)


def play(config, game='dnb', verbose=False):
    '''
    We'll probably want either a regular or command-line argument to choose 
    the trained network to play against.
    '''
    player_options = GAMES[game].PLAYER_TYPES
    player_options['alphazero'] = AlphaZero
    game = GAMES[game].Game(config)
    while not game.finished():
        if verbose or game.human_turn():
            game.display()
        game.play_turn()
    game.display()


if __name__ == '__main__':
    assert len(sys.argv) > 1
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = edict(yaml.safe_load(f))
    play(config)