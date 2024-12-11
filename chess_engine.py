import kagglehub
import chess
import pandas as pd
import io
import chess.pgn


def create_NN_input(game):

    for game in games:
        board = game.board() # create starting board
        print(type(board))
        
        print(board[0])
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