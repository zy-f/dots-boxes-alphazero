'''
file for representing game logic. defines:
- user-friendly representation/visualization of game state
- state and action representation
- state symmetry handling
- updating state based on moves
- any scoring heuristics
- determining and reporting the winner of a terminated game
'''

import numpy as np

LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

class DnBStr: # string representation definitions for dots and boxes board
    BLANK = '   '
    V_EDGE = ' | ' 
    H_EDGE = '---'
    DOT = ' o '

box_sides = {
    't': (-1, 0),
    'r': (0, 1),
    'b': (1, 0),
    'l': (0, -1)
}


class DnBBoard:
    ## STATIC FUNCTIONS
    def blank_board(nchar):
        board = [[DnBStr.BLANK] * nchar for _ in range(nchar)]
        chars = list(LETTERS[::-1])
        for i in range(nchar):
            for j in range(nchar):
                if i%2 == 0:
                    if j%2 == 0:
                        board[i][j] = DnBStr.DOT
                else:
                    if j%2  == 1:
                        board[i][j] = ' '+chars.pop()+' '
        return board
    
    def display_char(side_ix, value, ring_ix=0):
        if not value:
            return DnBStr.BLANK
        return DnBStr.H_EDGE if (side_ix % 2) == (ring_ix % 2) else DnBStr.V_EDGE
    
    def action_map(nb):
        mapping = []
        for ring_ix in range(nb):
            for side in range(4):
                mapping += [(ring_ix, side, i) for i in range(ring_ix+1)]
        return mapping
    
    def illegal_move_warning():
        print('move not legal, please try again')
    
    def board_from_rings(rings):
        nb = len(rings)
        nchar = 2*nb + 1
        board = DnBBoard.blank_board(nchar)
        for ring_ix, ring in enumerate(rings):
            sidelen = ring.shape[1]
            ix_start = nb - 1 - ring_ix
            ix_end = nchar - ix_start - 1 # to make it non-inclusive
            getchar = lambda v, side: DnBBoard.display_char(side, v, ring_ix=ring_ix)
            for i in range(sidelen):
                board[ix_start][ix_start+(i*2)+1] = getchar(ring[0, i], 0) # top edge
                board[ix_start+(i*2)+1][ix_end] = getchar(ring[1, i], 1) # right edge
                board[ix_end][ix_start+(i*2)+1] = getchar(ring[2, -(i+1)], 2) # bottom edge
                board[ix_start+(i*2)+1][ix_start] = getchar(ring[3, -(i+1)], 3) # left edge
        return board
    
    def rings_from_board(board):
        nchar = len(board)
        nb = nchar // 2
        rings = []
        for ring_ix in range(nb):
            ring = np.zeros((4,ring_ix+1), dtype=bool)
            sidelen = ring_ix+1
            ix_start = nb - 1 - ring_ix
            ix_end = nchar - ix_start - 1 # to make it non-inclusive
            for i in range(sidelen):
                ring[0, i] = (self.board[ix_start][ix_start+(i*2)+1] != DnBStr.BLANK) # top edge
                ring[1, i] = (self.board[ix_start+(i*2)+1][ix_end] != DnBStr.BLANK) # right edge
                ring[2, -(i+1)] = (self.board[ix_end][ix_start+(i*2)+1] != DnBStr.BLANK) # bottom edge
                ring[3, -(i+1)] = (self.board[ix_start+(i*2)+1][ix_start] != DnBStr.BLANK) # left edge
            rings.append(ring)
        return rings
    
    def str_repr_board(board):
        return '\n'.join([''.join(row) for row in board])
    ## STATIC FUNCTIONS END

    def __init__(self, num_boxes=3):
        self.nb = num_boxes
        self.nchar = 2 * self.nb + 1
        self.board = DnBBoard.blank_board(self.nchar)
        self.rings = [np.zeros((4,i+1)) for i in range(self.nb)] # innermost ring first
        self.action_mapping = DnBBoard.action_map(self.nb)
    
    def __str__(self):
        return DnBBoard.str_repr_board(self.board)
    
    def update_board_from_rings(self):
        self.board = DnBBoard.board_from_rings(self.rings)
    
    def update_rings_from_board(self):
        self.rings = DnBBoard.rings_from_board(self.board)
    
    def get_board_state(self):
        board_state = []
        for i in range(self.nb):
            row = []
            for j in range(self.nb):
                root_y, root_x = i*2+1, j*2+1
                row.append([
                    self.board[root_y-1][root_x] != DnBStr.BLANK, # top edge
                    self.board[root_y][root_x+1] != DnBStr.BLANK, # right edge
                    self.board[root_y+1][root_x] != DnBStr.BLANK, # bottom edge
                    self.board[root_y][root_x-1] != DnBStr.BLANK, # left edge
                ])
            board_state.append(row)
        return np.array(board_state, dtype=np.float32)
    
    def play(self, move): # returns True if move is legal, False otherwise
        if isinstance(move, tuple) or isinstance(move, list): # user-friendly input
            if len(move) != 2:
                DnBBoard.illegal_move_warning()
                return False
            box_label, side = move
            side = side.lower()
            box_number = LETTERS.find(box_label)
            if (side not in box_sides.keys()) or (box_number < 0 or box_number >= self.nb**2):
                DnBBoard.illegal_move_warning()
                return False
            dy, dx = box_sides[side]
            root_y, root_x = (box_number//self.nb)*2+1, (box_number%self.nb)*2+1
            self.board[root_y+dy][root_x+dx] = DnBStr.V_EDGE if side in 'lr' else DnBStr.H_EDGE
            self.update_rings_from_board()
        else: # numerical action index
            if (move < 0) or (move > len(self.action_mapping)):
                DnBBoard.illegal_move_warning()
                return False
            ring, side, sub_ix = self.action_mapping[move]
            self.rings[ring][side, sub_ix] = True
            self.update_board_from_rings()
        return True
    
    def legal_action_mask(self):
        flattened_rings = np.concatenate([r.flatten() for r in self.rings])
        return (flattened_rings == 0)
    
    def get_symmetries(self):
        # rotational symmetries
        index_rotations = [[i%4, (i+1)%4, (i+2)%4, (i+3)%4] for i in range(4)]
        symmetries = [[r[rot] for r in self.rings] for rot in index_rotations]
        # perform mirror image flip
        flipped = [r[:,::-1][[2,1,0,3]] for r in self.rings]
        symmetries += [[r[rot] for r in flipped] for rot in index_rotations]
        return symmetries
    
    def end_value(self):
        '''
        returns:
        - None if game is not yet complete
        - +1 if player 1 wins
        - -1 if player 2 wins
        '''
        pass


def debug():
    nb = 3
    board = DnBBoard(num_boxes=nb)
    move_space = nb*(nb+1) * 2
    moves = np.random.choice(move_space, size=10, replace=False)
    for move in moves:
        board.play(move)
        print('\n')
        print(board)
    print('\n\n==========\n\n')
    for sym in board.get_symmetries():
        print(DnBBoard.str_repr_board(DnBBoard.board_from_rings(sym)))
        print('\n')
    # while 1:
    #     inp = input("play: ").strip()
    #     if inp.lower() == 'q':
    #         break
    #     try:
    #         move = int(inp)
    #     except:
    #         move = inp.split()
    #     board.play(move)
    #     print(board)
    #     print(board.legal_action_mask())


if __name__ == "__main__":
    debug()