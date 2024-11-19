'''
implements the Monte-Carlo Tree Search step of the algorithm
i'm doing this i guess
'''


def search(board, net, n_sim=100, c_puct=1, noise=.03, tau_pi=1, verbose=False):
    '''
    inputs:
    - board = a board (class instance, not just the representation)
    - net = the neural network that returns policies and values
    - n_sim = number of rollouts to perform
    - c_puct = exploration vs exploitation parameter
    - noise = dirichlet noise added to move probabilities
    - tau_pi = standard temperature param for softening probabilities in pi

    returns move probabilities pi across all actions for the next move
    '''
    pi = None
    return pi
