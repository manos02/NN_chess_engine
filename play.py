from model import ChessModel
from state import State
import torch
import sys
import chess



def load_model(path):
    try:
        vals = torch.load(path, weights_only=True)
        model = ChessModel()
        model.load_state_dict(vals)
        return model
    except Exception as e:
        print(f"Error loading AI model: {e}")
        sys.exit()


def evaluate(s, model):
    
    b = s.board_to_matrix()
    input_tensor = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    
    return output.item()


def human_move(selected_square, square, s, ai_thinking):
    if selected_square is None:
        piece = s.board.piece_at(square)
        if piece and piece.color == chess.BLACK:
            selected_square = square
    else:
        move = chess.Move(selected_square, square)
        
        if move in s.board.legal_moves:
            s.board.push(move)
            selected_square = None
            ai_thinking = True
        else:
            print("Invalid move")
            selected_square = None
    return s, selected_square, ai_thinking


# int alphaBetaMax( int alpha, int beta, int depthleft ) {
#    if ( depthleft == 0 ) return evaluate();
#    bestValue = -infinity;
#    for ( all moves) {
#       score = alphaBetaMin( alpha, beta, depthleft - 1 );
#       if( score > bestValue )
#       {
#          bestValue = alpha;
#          if( score > alpha )
#             alpha = score; // alpha acts like max in MiniMax
#       }
#       if( score >= beta )
#          return score;   // fail soft beta-cutoff
#    }
#    return bestValue;
# }

def alphaBetaMax(depth, s, alpha, beta, maxPlayer, model):
    if depth == 0 or s.board.is_game_over():
        return evaluate(s, model), None

    bestMove = None
    if maxPlayer: 
        bestScore = -float('inf')
        for move in s.board.legal_moves:
            s.board.push(move)
            score, m = alphaBetaMax(depth-1, s, alpha, beta, False, model)
            s.board.pop()
            
            if score > bestScore:
                bestScore = score   
                bestMove = move         
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return bestScore, bestMove

    else:
        bestScore = float('inf')
        for move in s.board.legal_moves:

            s.board.push(move)
            score, m = alphaBetaMax(depth-1, s, alpha, beta, True, model)
            s.board.pop()
            
            if score < bestScore:
                bestScore = score
                bestMove = move
            if score < beta:
                beta = score
            if alpha >= beta:
                break

        return bestScore, bestMove

    
    

