"""
Tic Tac Toe Player
"""

import math
import copy
from exceptions import InvalidMove

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    count_x, count_o = 0, 0
    for row in board:
        for cell in row:
            if cell is X:
                count_x += 1
            elif cell is O:
                count_o += 1
    
    if count_x > count_o:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_move = set()
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if cell is EMPTY:
                possible_move.add((i, j))
    return possible_move


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if terminal(board) or action not in actions(board):
        raise InvalidMove("Not a valid move")

    new_board = copy.deepcopy(board)
    turn = player(board)
    
    new_board[action[0]][action[1]] = turn
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # check horizontal win
    for row in board:
        if row[0] == row[1] and row[1] == row[2]:
            if row[0] is not EMPTY:
                return row[0]

    # check vertical win
    for i in range(3):
        if board[0][i] == board[1][i] and board[1][i] == board[2][i]:
            if board[0][i] is not EMPTY:
                return board[0][i]
    
    # check diagonal win
    if (board[0][0] == board[1][1] and board[1][1] == board[2][2]) or (board[0][2] == board[1][1] and board[1][1] == board[2][0]):
        if board[1][1] is not EMPTY:
            return board[1][1]
    
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # check whether there is a winner
    if winner(board) is not None:
        return True
    
    # check whether all cells have been filled up
    for row in board:
        for cell in row:
            if cell is EMPTY:
                return False
    
    # if there is no winner and all cells have been filled up, then draw state
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    end_state = terminal(board)
    if end_state:
        w = winner(board)
        if w is X:
            return 1
        elif w is O:
            return -1
        else:
            return 0 


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # Min player wants to minimize max player utility
    def min_value(board):
        if terminal(board):
            return utility(board)
        v = 1000
        for action in actions(board):
            v = min(v, max_value(result(board, action)))
        return v

    # Max player wants to maximize its own utility
    def max_value(board):
        if terminal(board):
            return utility(board)
        v = -1000
        for action in actions(board):
            v = max(v, min_value(result(board, action)))
        return v
    
    p = player(board)

    if board == initial_state():
        return (1, 1)
    
    if terminal(board):
        return None

    # if player is X
    if p is X:
        best_utility = -1000
        move = None
        for action in actions(board):
            min_utility = min_value(result(board, action))
            if min_utility > best_utility:
                best_utility = min_utility
                move = action
    # if player is O
    else:
        best_utility = 1000
        move = None
        for action in actions(board):
            max_utility = max_value(result(board, action))
            if max_utility < best_utility:
                best_utility = max_utility
                move = action
    return move
