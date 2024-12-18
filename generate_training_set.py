import pandas as pd
import chess
import numpy as np
import io
import chess.pgn
from state import State
import torch


def create_NN_input(games):

    X = []
    y = []

    for i, game in enumerate(games):

        board = game.board() # create starting board

        value = {'1-0':1,'1/2-1/2':0,'0-1':-1}[game.headers['Result']] # 1 for white win, 0-0 for black win, -1 for black win
    
        for move in game.mainline_moves(): # for every move
            X.append(State(board).board_to_matrix())
            y.append(value) # result of the game
            board.push(move) # execute the move
            print(f"parsing game: {i}, sample: {len(X)}")
                        
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

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X,y


if __name__ == "__main__":
  X,y = get_dataset(100)
  np.savez_compressed("dataset_100.npz", X, y)