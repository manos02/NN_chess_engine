from model import ChessModel
import chess
import chess.svg
from state import State
import torch

PATH = "nets/value.pth"


if __name__ == "__main__":



    vals = torch.load(PATH, weights_only=True)
    model = ChessModel()
    model.load_state_dict(vals)

    s = State()

        

    while not s.board.is_game_over():
        print("Move from: ", end="")
        from_square = input()
        print("To: ", end="")
        to_square = input()
        try:
            move = chess.Move.from_uci(from_square+to_square)
        except:
            print("The move is not valid") # if the squares are wrong
            continue

        if move in s.board.legal_moves:
            s.board.push(move)
        else:
            print("The move is not legal") # if the move is not valid
            continue

        print(s.board)
        print()

        chess_moves = {}
        for move in s.board.legal_moves: # turn of AI
            s.board.push(move)
            b = s.board_to_matrix()
            input_tensor = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
            output = model(input_tensor)
            chess_moves[move] = output
            s.board.pop()

        best_move = min(chess_moves, key = chess_moves.get)
        s.board.push(best_move)
        print(chess_moves[best_move])
        print(s.board)

        

    
