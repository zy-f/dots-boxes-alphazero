import numpy as np
from dots_boxes.game_logic import DnBBoard
from dots_boxes.nnet import DnBNet

PLAYER_TYPES = {}

class Player:
    is_human = False
    label = "null"
    def move(self, board):
        pass

class Human(Player):
    is_human = True
    label = "Human"
    def move(self, board):
        inp = input("Your move: ").strip()
        return inp.split()
PLAYER_TYPES['human'] = Human

class BaselineBot(Player):
    label = "Baseline (bot)"
    def move(self, board):
        legal_moves = np.where(board.legal_action_mask())[0]
        scoring_moves = []
        neutral_moves = []
        for a in legal_moves:
            scratch = board.clone()
            scratch.play(a)
            if sum(scratch.scores) > sum(board.scores):
                scoring_moves.append(a)
            else: # make sure you didn't set up the opponent to score
                state, _ = scratch.nn_state()
                n_three_edges = (state.sum(axis=2) == 3).sum()
                if n_three_edges < 1:
                    neutral_moves.append(a)
        if len(scoring_moves) > 0:
            move = np.random.choice(scoring_moves)
        elif len(neutral_moves) > 0:
            move = np.random.choice(neutral_moves)
        else:
            move = np.random.choice(legal_moves)
        return move
PLAYER_TYPES['baseline'] = BaselineBot

def makesize(size):
    try:
        size = int(size)
        if size < 2 or size > 7:
            size = -1
    except:
        size = -1
    return size


class DnBGame:
    def __init__(self, config, player_opts=PLAYER_TYPES):
        print("Let's play dots and boxes!")
        print("Player options: " + ', '.join([f'"{p}"' for p in player_opts]))
        p1 = p2 = None
        while p1 not in player_opts:
            if p1 is not None:
                print("Invalid option, please try again")
            p1 = input("Player 1: ")
        while p2 not in player_opts:
            if p2 is not None:
                print("Invalid option, please try again")
            p2 = input("Player 2: ")
        board_size = -1
        while board_size < 1:
            board_size = makesize(input("Board size [2-7]: "))
        self.board = DnBBoard(num_boxes=board_size)
        
        if p1 == "alphazero":
            self.players = [player_opts[p1](config), player_opts[p2]()]
        elif p2 == "alphazero":
            self.players = [player_opts[p1](), player_opts[p2](config)]
        else:
            self.players = [player_opts[p1](), player_opts[p2]()]
            
        print('--- Instructions ---')
        print("Move format: <box label> [space] <side ([t]op/[b]ottom/[l]eft/[r]ight)>")
        print("For example, `a b` would select the bottom edge of box a")
        print('--- Game Start ---')
    
    def display(self):
        print(self.board)
        print(f"SCORE: {self.players[0].label} {self.board.scores[0]} - {self.board.scores[1]} {self.players[1].label}")
    
    def play_turn(self):
        player = self.players[self.board.player_turn]
        errs = 0
        while errs < 10:
            move = player.move(self.board)
            if self.board.play(move):
                return
            errs += 1
        raise ValueError

    def human_turn(self):
        return self.players[self.board.player_turn].is_human
    
    def finished(self):
        return self.board.end_value() is not None