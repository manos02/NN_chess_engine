import kagglehub
import chess
import pandas as pd
import io
import chess.pgn
import numpy as np
import torch
import time
from torch import optim
from torch.utils.data import DataLoader 
from model import ChessModel
from chess_dataset import ChessDataset
from tqdm import tqdm 

def board_to_matrix(board:chess.Board):
    
    ''' 
    Create a matrix with 13 boards: 6 for the white pieces, 6 for the black pieces, 1 for the available moves
    '''
    matrix = np.zeros((13, 8, 8))

    # piece map returns a dictionary with key the square as int and the value if the Piece object, h8 -> 63, a1 -> 0 and so on
    piece_map = board.piece_map() 
    
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
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        
        
        square = move.to_square # legal square to move
        row = square // 8
        col = square % 8
        matrix[12][row][col] = 1

    return matrix


def encode_moves(moves):
    # create a dict with move as the key, and an int as the value e.g np.str_('c5c7'): 0
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    
    # return the value for each move and the dictionary with the encoded moves
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int


def create_NN_input(game):

    X = []
    y = []

    for game in games:
        board = game.board() # create starting board
    

        for move in game.mainline_moves(): # for every move
            X.append(board_to_matrix(board))
            y.append(move.uci()) # move played
            board.push(move) # execute the move
                        
    return np.array(X), np.array(y)




if __name__ == '__main__':

    path = kagglehub.dataset_download("arevel/chess-games")
    data = pd.read_csv("chess_games.csv", nrows=5000) # 10 for testing, change later
    data = data[['White', 'Black', 'Result', 'WhiteElo', 'BlackElo', 'Termination', 'AN']]

    games = []

    for row in data.loc[:, 'AN']: # iterate games column
    
        pgn = io.StringIO(row)
        game = chess.pgn.read_game(pgn)
        games.append(game)
            
    # X position of the board during the game
    # y next move from that position
    X, y = create_NN_input(games)

    
    y, move_to_int = encode_moves(y)
    
    num_classes = len(move_to_int)
    

    print(f"NUMBER OF SAMPLES: {len(y)}")


    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    print(y[0])
    print(move_to_int)


    # check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = ChessModel(num_classes=num_classes).to(device)
    
    dataset = ChessDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 50
    for epoch in range(epochs):  # loop over the dataset multiple times
        start_time = time.time()
        running_loss = 0.0
        model.train()

        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels) # calculate loss
            loss.backward() # backward pass

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()

        end_time = time.time()
        epoch_time = end_time - start_time
        minutes: int = int(epoch_time // 60)
        seconds: int = int(epoch_time) - minutes * 60
        print(f'Epoch {epoch + 1 + 50}/{epochs + 1 + 50}, Loss: {running_loss / len(dataloader):.4f}, Time: {minutes}m{seconds}s')
        


