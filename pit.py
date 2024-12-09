import dots_boxes
from mcts import MCTS
import numpy as np
from dots_boxes.nnet import *
from play import *
from dots_boxes.play_logic import *
import sys
import yaml
from easydict import EasyDict as edict

if __name__ == '__main__':
    assert len(sys.argv) == 4
    board_size = int(sys.argv[1])
    player1 = sys.argv[2]
    player2 = sys.argv[3]

    if player1 == "random":
        player1 = RandomBaselineBot()
    elif player1 == "greedy":
        player1 = GreedyBaselineBot()
    else:
        with open(player1, 'r') as f:
          player1 = edict(yaml.safe_load(f))
        player1 = AlphaZero(player1)
    
    if player2 == "random":
        player2 = RandomBaselineBot()
    elif player2 == "greedy":
        player2 = GreedyBaselineBot()
    else:
        with open(player2, 'r') as f:
            player2 = edict(yaml.safe_load(f))
        player2 = AlphaZero(player2)

    players = [player1, player2]
    
    num_games = 32
    total_wins = 0.0
    for i in range(num_games // 2):
        board = DnBBoard(num_boxes=board_size)
        while board.end_value() is None:
            player = players[board.player_turn]
            move = player.move(board)
            board.play(move)
        winner = board.end_value()
        if winner == 0:
            total_wins += 0.5
        if winner == +1:
            total_wins += 1.0
    
    players = players[::-1]
    for i in range(num_games // 2):
        board = DnBBoard(num_boxes=board_size)
        while board.end_value() is None:
            player = players[board.player_turn]
            move = player.move(board)
            board.play(move)
        winner = board.end_value()
        if winner == 0:
            total_wins += 0.5
        if winner == -1:
            total_wins += 1.0
    
    print(f"Player 1 win rate is {total_wins / num_games:.3f}.")
        

    
