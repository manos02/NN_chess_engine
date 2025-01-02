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


def human_move(selected_square, square, s):
    if selected_square is None:
        piece = s.board.piece_at(square)
        if piece and piece.color == chess.BLACK:
            selected_square = square
        return s, selected_square, False, None 
    else:
        move = chess.Move(selected_square, square)       
        if chess.square_rank(square) == 0 and s.board.piece_at(selected_square).symbol() == 'p': # if promotion square
            pieces_to_nums = {"q":5, "k":2, "r":4, "b":3}
            while True:
                print("Promote to: q (Queen), k (Knight), b (Bishop), r (Rook)")
                ans = input()
                if ans in "qkbr":
                    move = chess.Move(selected_square, square, pieces_to_nums[ans])
                    print(move)
                    break

        if move in s.board.legal_moves:
            s.board.push(move)
            return s, None, True, move
        else:
            print("Invalid move")
            return s, None, False, None


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

    
    

