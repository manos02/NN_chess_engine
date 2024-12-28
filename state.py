import chess
import numpy as np


class State():
    
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def board_to_matrix(self):
        ''' 
        Create a matrix with 14 boards:
        6 for the white pieces, 6 for the black pieces
        13th board:
        1 for the available moves, 1 for colour to play
        2 white castling and 2 black castling
        '''
        matrix = np.zeros((14, 8, 8))

        # piece map returns a dictionary with key the square as int and the value if the Piece object, h8 -> 63, a1 -> 0 and so on
        piece_map = self.board.piece_map() 
        
        for square, piece in piece_map.items():
            row = square // 8
            col = square % 8

            # get the piece, 1:pawn, 2:bishop, 3:knight, 4:rook, 5:queen, 6:king
            p = piece.piece_type - 1
            
            # False for white, True for black
            p_color = piece.color

            # append to the correct board
            matrix[p + 0 if p_color else 6, row, col] = 1

        # get legal moves, 13th board
        legal_moves = list(self.board.legal_moves)
        for move in legal_moves:

            square = move.to_square # legal square to move
            row = square // 8
            col = square % 8
            matrix[12][row][col] = 1

        # 14th board

        # 1st col white's turn set the first column to 1
        if self.board.turn: 
            matrix[13, :, 0] = 1
        
        # 2nd col white king castling
        if self.board.has_kingside_castling_rights(chess.WHITE):
            matrix[13, :, 1] = 1

        # 3nd col white queen castling
        if self.board.has_queenside_castling_rights(chess.WHITE):
            matrix[13, :, 2] = 1

        # 4nd col white king castling
        if self.board.has_kingside_castling_rights(chess.BLACK):
            matrix[13, :, 3] = 1

        # 5th col white king castling
        if self.board.has_queenside_castling_rights(chess.BLACK):
            matrix[13, :, 4] = 1

        # 6th col represent en peasant
        if self.board.has_legal_en_passant:
            matrix[13, :, 5] = 1
            
        # 7th col representing if it is a check
        if self.board.is_check:
            matrix[13, :, 7] = 1

                

        return matrix


