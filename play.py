'''
file to allow interaction between user and a trained alphazero instance
basically the inference-time file that lets you play vs the bot user
'''
import dots_boxes
from mcts import MCTS
import numpy as np

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
    def __init__(self, game='dnb'):
        self.net = GAMES[game].Network(pretrained=True)
        self.mcts = MCTS(params={'tau_pi': 0})

    def move(self, board):
        policy = self.mcts.search(board, self.net)
        return np.argmax(policy)


def play(game='dnb', verbose=False):
    '''
    We'll probably want either a regular or command-line argument to choose 
    the trained network to play against.
    '''

    game = GAMES[game].Game()
    while not game.finished():
        if verbose or game.human_turn():
            game.display()
        game.play_turn()
    game.display()


if __name__ == '__main__':
    play()