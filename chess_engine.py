import kagglehub
import chess
import pandas as pd
import io
import chess.pgn
import numpy as np

def board_to_matrix(board:chess.Board):
    
    ''' 
    Create a matrix with 13 boards: 6 for the white pieces, 6 for the black pieces, 1 for all pieces, 1 for the available moves
    '''
    matrix = np.zeros((13, 8, 8))

    # piece map returns a dictionary with key the square as int and the value if the Piece object, h8 -> 63, a1 -> 0 and so on
    piece_map = board.piece_map() 
    
    for square, piece in piece_map.items():
        row = square // 8
        col = square % 8
        
        print(square, row, col)

        # get the piece, 1:pawn, 2:bishop, 3:knight, 4:rook, 5:queen, 6:king
        p = piece.piece_type - 1
        
        # False for white, True for black
        p_color = piece.color

        # append to the correct board
        matrix[p + 0 if p_color else 6, row, col] = 1
        


def create_NN_input(game):

    X = []
    y = []

    for game in games:
        print(game)
        board = game.board() # create starting board
        # print(type(board))
        for move in game.mainline_moves(): # for every move
            X.append(board_to_matrix(board))
            board.push(move)
            
            
        
        # print(board)
        break
        
    
    return 1,2





if __name__ == '__main__':

    path = kagglehub.dataset_download("arevel/chess-games")
    data = pd.read_csv("chess_games.csv", nrows=500)
    data = data[['White', 'Black', 'Result', 'WhiteElo', 'BlackElo', 'Termination', 'AN']]

    games = []

    for row in data.loc[:, 'AN']:
    
        pgn = io.StringIO(row)
        game = chess.pgn.read_game(pgn)
        games.append(game)
        
    
    
    
    # X position of the board during the game
    # y next move from that position
    X, y = create_NN_input(games) 





    board = chess.Board()
    # print(board.legal_moves)