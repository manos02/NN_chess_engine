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

# Well above any material/model score, so a mate always outweighs material
MATE_SCORE = 10000

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

def handcrafted_evaluate(s):
    score = 0
    for piece_type in PIECE_VALUES:
        score += len(s.board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        score -= len(s.board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
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


# MVV-LVA: grabbing a big piece with a small one is the most promising try, so
# searching those first makes alpha-beta cut off far more of the tree.
def move_score(board, move):
    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        # en passant leaves the target square empty; the victim is always a pawn
        victim_value = PIECE_VALUES[victim.piece_type] if victim else PIECE_VALUES[chess.PAWN]
        attacker_value = PIECE_VALUES[board.piece_at(move.from_square).piece_type]
        score += 10 * victim_value - attacker_value + 100
    if move.promotion:
        score += PIECE_VALUES[move.promotion]
    return score

def ordered_moves(board, captures_only=False):
    moves = [m for m in board.legal_moves if not captures_only or board.is_capture(m)]
    moves.sort(key=lambda m: move_score(board, m), reverse=True)
    return moves


# Keep searching captures past the nominal depth until the position is quiet, so
# leaves are never scored in the middle of a trade (the horizon effect).
def quiesce(s, alpha, beta, maxPlayer, model):
    if s.board.is_checkmate():
        score = MATE_SCORE
        return -score if s.board.turn == chess.WHITE else score
    if s.board.is_stalemate():
        return 0

    # The side to move is not obliged to capture, so its static score is a floor
    # (a ceiling when minimizing) that any capture has to beat.
    stand_pat = combined_evaluate(s, model)
    if maxPlayer:
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha:
            return stand_pat
        beta = min(beta, stand_pat)

    for move in ordered_moves(s.board, captures_only=True):
        s.board.push(move)
        score = quiesce(s, alpha, beta, not maxPlayer, model)
        s.board.pop()

        if maxPlayer:
            alpha = max(alpha, score)
        else:
            beta = min(beta, score)
        if alpha >= beta:
            break

    return alpha if maxPlayer else beta


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
def alphaBetaMax(depth, s, alpha, beta, maxPlayer, model, is_root=False):
    # Score a repeated position as a draw so the engine stops shuffling. Skipped at
    # the root, which must always return a move, and when no reversible move has
    # been made yet (a repetition is impossible then).
    if not is_root and s.board.halfmove_clock >= 4:
        if s.board.is_repetition(2) or s.board.can_claim_fifty_moves():
            return 0, None

    if s.board.is_game_over():
        if s.board.is_checkmate():
            # White-relative, like the rest of the evaluation. `depth` is the depth
            # left to search, so shallower mates score higher and get played first.
            score = MATE_SCORE + depth
            return -score if s.board.turn == chess.WHITE else score, None
        return 0, None  # stalemate, insufficient material, fivefold, 75-move

    if depth == 0:
        return quiesce(s, alpha, beta, maxPlayer, model), None

    bestMove = None
    if maxPlayer:
        bestScore = -float('inf')
        for move in ordered_moves(s.board):
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
        for move in ordered_moves(s.board):
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

    
    

