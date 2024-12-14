import pandas as pd
import chess
import numpy as np
import io
import chess.pgn
from state import State



def create_NN_input(games):

    X = []
    y = []

    for game in games:
        board = game.board() # create starting board
    
        for move in game.mainline_moves(): # for every move
            X.append(State(board).board_to_matrix())
            y.append(move.uci()) # move played
            board.push(move) # execute the move
                        
    return np.array(X), np.array(y)
                        

def get_dataset(num_games=None):
    # path = kagglehub.dataset_download("arevel/chess-games")

    data = pd.read_csv("chess_games.csv", nrows=num_games)  
    data = data[['White', 'Black', 'Result', 'WhiteElo', 'BlackElo', 'Termination', 'AN']]
    games = []

    for row in data.loc[:, 'AN']: # iterate games column
        
        pgn = io.StringIO(row)
        game = chess.pgn.read_game(pgn)
        games.append(game)

    # X position of the board during the game
    # y next move from that position
    X, y = create_NN_input(games)

    # X = torch.tensor(X, dtype=torch.float32)
    # y = torch.tensor(y, dtype=torch.long)

    return X,y


if __name__ == "__main__":
  X,y = get_dataset(100)
  np.savez("dataset_25M.npz", X, y)