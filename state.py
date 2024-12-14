import pandas as pd
import numpy as np



def encode_moves(moves):
    # create a 
    #  with move as the key, and an int as the value e.g np.str_('c5c7'): 0
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    
    # return the value for each move and the dictionary with the encoded moves
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int

class State():
    def __init__(self, board):
        self.board = board


    def board_to_matrix(self):
        ''' 
        Create a matrix with 13 boards: 6 for the white pieces, 6 for the black pieces, 1 for the available moves
        '''
        matrix = np.zeros((13, 8, 8))

        # piece map returns a dictionary with key the square as int and the value if the Piece object, h8 -> 63, a1 -> 0 and so on
        piece_map = self.board.piece_map() 
        
        for square, piece in piece_map.items():
            row = square // 8
            col = square % 8
            
            # print(square, row, col)

            # get the piece, 1:pawn, 2:bishop, 3:knight, 4:rook, 5:queen, 6:king
            p = piece.piece_type - 1
            
            # False for white, True for black
            p_color = piece.color

            # append to the correct board
            matrix[p + 0 if p_color else 6, row, col] = 1

        # get legal moves
        legal_moves = list(self.board.legal_moves)
        for move in legal_moves:

            square = move.to_square # legal square to move
            row = square // 8
            col = square % 8
            matrix[12][row][col] = 1

        return matrix


