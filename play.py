from model import ChessModel
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

def handcrafted_evaluate(s):

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  
    }
    
    score = 0
    for piece_type in piece_values:
        score += len(s.board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        score -= len(s.board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    return score

def model_evaluate(s, model):
    b = s.board_to_matrix()
    input_tensor = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    
    return output.item()



def combined_evaluate(s, model):

    model_score = model_evaluate(s, model)
    handcrafted_score = handcrafted_evaluate(s)
    
    combined_score = model_score + handcrafted_score
    return combined_score


# Returns (state, selected_square, ai_to_move, move, needs_promotion).
# needs_promotion is True when the click is a legal promotion but no piece was
# chosen yet; the caller should ask the user and call again with `promotion`.
def human_move(selected_square, square, s, human_color=chess.BLACK, promotion=None):
    if selected_square is None:
        piece = s.board.piece_at(square)
        if piece and piece.color == human_color:
            selected_square = square
        return s, selected_square, False, None, False
    else:
        move = chess.Move(selected_square, square)
        promotion_rank = 7 if human_color == chess.WHITE else 0
        piece = s.board.piece_at(selected_square)
        if chess.square_rank(square) == promotion_rank and piece is not None and piece.piece_type == chess.PAWN: # if promotion square
            if promotion is None:
                if chess.Move(selected_square, square, chess.QUEEN) in s.board.legal_moves:
                    return s, selected_square, False, None, True
            else:
                move = chess.Move(selected_square, square, promotion)

        if move in s.board.legal_moves:
            s.board.push(move)
            return s, None, True, move, False
        else:
            print("Invalid move")
            return s, None, False, None, False


# https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
def alphaBetaMax(depth, s, alpha, beta, maxPlayer, model):
    if depth == 0 or s.board.is_game_over():
        return combined_evaluate(s, model), None

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

    
    

