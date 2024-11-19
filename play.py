'''
file to allow interaction between user and a trained alphazero instance
basically the inference-time file that lets you play vs the bot user
'''
from dots_boxes.nnet import DnBNet
from dots_boxes.game_logic import DnBBoard

def request_user_move():
    legal_move = False
    while not legal_move:
        inp = input("Your move: ").strip()
        move = inp.split()
        legal_move = board.play(move)

def dnb_instructions():
    print("Let's play dots and boxes!")
    print("Move format: <box label> [space] <side ([t]op/[b]ottom/[l]eft/[r]ight)>")
    print("For example, `a b` would select the bottom edge of box a")

def turn_order():
    inp = input("Would you like to go first (y/n)? ")
    user_first = inp.lower().startswith('y')
    return user_first

def play():
    '''
    We'll probably want either a regular or command-line argument to choose 
    the trained network to play against.
    '''
    dnb_instructions()
    user_first = turn_order()
    pass



if __name__ == '__main__':
    play_dnb()