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
    DOT = ' • '

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
    
    def action_map(nb):
        mapping = []
        for ring_ix in range(nb):
            for side in range(4):
                mapping += [(ring_ix, side, i) for i in range(ring_ix+1)]
        return mapping
    
    def illegal_move_warning():
        print('move not legal, please try again')
    
    def ring_board_pairs(nb):
        nchar = 2*nb + 1
        pairs = []
        for r_ix in range(nb):
            sidelen = r_ix + 1
            ix_start = nb - sidelen
            ix_end = nchar - ix_start - 1
            for i in range(sidelen):
                pairs += [
                    ((ix_start, ix_start+(i*2)+1), (r_ix, 0, i)),
                    ((ix_start+(i*2)+1, ix_end), (r_ix, 1, i)),
                    ((ix_end, ix_start+(i*2)+1), (r_ix, 2, sidelen-i-1)),
                    ((ix_start+(i*2)+1, ix_start), (r_ix, 3, sidelen-i-1))
                ]
        return pairs
    
    def board_from_rings(rings):
        nb = len(rings)
        nchar = 2*nb + 1
        board = DnBBoard.blank_board(nchar)
        for ring_ix, ring in enumerate(rings):
            sidelen = ring.shape[1]
            ix_start = nb - 1 - ring_ix
            ix_end = nchar - ix_start - 1 # to make it non-inclusive
            getchar = lambda v, side: DnBBoard.display_char(nb, side, v, ring_ix=ring_ix)
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
    
    def display_char(nb, side_ix, value, ring_ix=0):
        if not value:
            return DnBStr.BLANK
        is_hz = (side_ix % 2) != ((nb - ring_ix) % 2)
        return DnBStr.H_EDGE if is_hz else DnBStr.V_EDGE
    
    def str_repr_board(board):
        return '\n'.join([''.join(row) for row in board])
    
    def get_symmetrical_states(tree_state):
        # make rings
        ring_state, scores = tree_state
        flat_rings = [int(c) for c in ring_state]
        nb = int(np.sqrt(len(flat_rings) // 2))
        rings = []
        for i in range(nb):
            rings.append(np.array(flat_rings[:4*(i+1)]).reshape(4,i+1))
            flat_rings = flat_rings[4*(i+1):]
        # rotational symmetries
        index_rotations = [[i%4, (i+1)%4, (i+2)%4, (i+3)%4] for i in range(4)]
        symmetries = [[r[rot] for r in rings] for rot in index_rotations]
        # perform mirror image flip
        flipped = [r[:,::-1][[2,1,0,3]] for r in rings]
        symmetries += [[r[rot] for r in flipped] for rot in index_rotations]
        # map symmetrical rings to equivalent tree states
        sym_states = []
        for sym in symmetries:
            flat = np.concatenate([r.flatten() for r in sym]).astype(int)
            sym_states.append((''.join([str(x) for x in flat]), scores))
        return sym_states

    ## STATIC FUNCTIONS END

    def __init__(self, num_boxes=3, tree_state=None):
        if tree_state is not None:
            ring_state, scores = tree_state
            flat_rings = [int(c) for c in ring_state]
            # recall that (# elems) = 2 * n(n+1) because it's 4 * (sequential sum)
            self.nb = int(np.sqrt(len(flat_rings) // 2))
            self.nchar = 2 * self.nb + 1
            self.player_turn = sum(flat_rings) % 2
            # make rings
            self.rings = []
            for i in range(self.nb):
                self.rings.append(np.array(flat_rings[:4*(i+1)]).reshape(4,i+1))
                flat_rings = flat_rings[4*(i+1):]
            self.board = DnBBoard.board_from_rings(self.rings)
            self.scores = list(scores)
        else:
            self.nb = num_boxes
            self.nchar = 2 * self.nb + 1
            self.board = DnBBoard.blank_board(self.nchar)
            self.rings = [np.zeros((4,i+1)) for i in range(self.nb)] # innermost ring first
            self.scores = [0,0] # number of boxes per player
            self.player_turn = 0 # 0 or 1
        self.action_mapping = DnBBoard.action_map(self.nb)
        edge_pairs = DnBBoard.ring_board_pairs(self.nb)
        self.board2ring = {b_ix: r_ix for (b_ix, r_ix) in edge_pairs} 
        self.ring2board = {r_ix: b_ix for (b_ix, r_ix) in edge_pairs}
    
    def clone(self):
        return self.__class__(tree_state=self.tree_state())
    
    def __str__(self):
        return DnBBoard.str_repr_board(self.board)
    
    def tree_state(self):
        flattened_rings = np.concatenate([r.flatten() for r in self.rings]).astype(int)
        return ''.join([str(x) for x in flattened_rings]), tuple(self.scores)
    
    def nn_state(self):
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
        return np.array(board_state, dtype=np.float32), tuple(self.scores)
    
    def update_scores(self):
        scores_updated = False
        state, _ = self.nn_state()
        n_boxes = (state.sum(axis=2) == 4).sum()
        if n_boxes > sum(self.scores):
            self.scores[self.player_turn] += 1
            scores_updated = True
        return scores_updated
    
    def play(self, move): # returns True if move is legal, False otherwise

        # TODO: does not currently handle moves on occupied sides yet

        y, x = (None, None)
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
            y = root_y+dy
            x = root_x+dx
            ring, side, sub_ix = self.board2ring[(root_y+dy, root_x+dx)]
        else: # numerical action index
            if (move < 0) or (move >= len(self.action_mapping)):
                DnBBoard.illegal_move_warning()
                return False
            ring, side, sub_ix = self.action_mapping[move]
            y, x = self.ring2board[(ring, side, sub_ix)]
        if self.rings[ring][side, sub_ix]:
            DnBBoard.illegal_move_warning()
            return False
        self.rings[ring][side, sub_ix] = True
        self.board[y][x] = DnBBoard.display_char(self.nb, side, True, ring_ix=ring)
        scores_updated = self.update_scores()
        # by convention, we switch to the other player's turn when the game ends
        # this is to interface properly with mcts in the general case
        if (self.end_value() is not None) or (not scores_updated):
            self.player_turn = int(not self.player_turn) # change turn
        return True
    
    def legal_action_mask(self):
        flattened_rings = np.concatenate([r.flatten() for r in self.rings]).astype(int)
        return (flattened_rings == 0)
    
    def end_value(self):
        '''
        returns:
        - None if game is not yet complete
        - +1 if player 1 wins
        - -1 if player 2 wins
        - +0 if draw
        '''
        diff = self.scores[0] - self.scores[1]
        return None if sum(self.scores) < self.nb**2 else np.sign(diff).item()


def debug():
    nb = 2
    board = DnBBoard(num_boxes=nb)
    print(board)
    # move_space = nb*(nb+1) * 2
    # moves = np.random.choice(move_space, size=10, replace=False)
    # for move in moves:
    #     board.play(move)
    #     print('\n')
    #     print(board)
    # print('\n\n==========\n\n')
    # for sym in board.get_symmetries():
    #     print(DnBBoard.str_repr_board(DnBBoard.board_from_rings(sym)))
    #     print('\n')
    while 1:
        print(board.tree_state())
        print(board.nn_state())
        print(board.action_mapping)
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