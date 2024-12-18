import pandas as pd
import chess
import numpy as np
import io
import chess.pgn
from state import State
import torch


def create_NN_input(games, target_num_samples):

    X = []
    y = []
    num_samples = 0


    for i, game in enumerate(games):

        board = game.board() # create starting board

        value = {'1-0':1,'1/2-1/2':0,'0-1':-1}[game.headers['Result']] # 1 for white win, 0-0 for black win, -1 for black win
        for move in game.mainline_moves(): # for every move
            num_samples += 1

            X.append(State(board).board_to_matrix())
            y.append(value) # result of the game
            board.push(move) # execute the move
            
            print(f"parsing game: {i}, sample: {num_samples}")

        if num_samples >= target_num_samples:
            break
                        
    return np.array(X), np.array(y)
                        

def get_dataset(target_num_samples=100000):

    '''
    Data fron https://www.kaggle.com/datasets/arevel/chess-games
    Around 6 million games
    '''

    data = pd.read_csv("chess_games.csv", nrows=10000) # change rows later  
    data = data[['White', 'Black', 'Result', 'WhiteElo', 'BlackElo', 'Termination', 'AN']]
    games = []

    
    for row in data.loc[:, 'AN']: # iterate games column
        
        pgn = io.StringIO(row)
        game = chess.pgn.read_game(pgn)
        games.append(game)

    # X position of the board during the game
    # y next move from that position
    X, y = create_NN_input(games, target_num_samples)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X,y


if __name__ == "__main__":
  X,y = get_dataset() # pass the number of samples
  np.savez_compressed("dataset_100.npz", X, y)