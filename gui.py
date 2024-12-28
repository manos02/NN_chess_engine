import pygame
from state import State
import sys
import chess
import torch
from model import ChessModel


# pygame setup
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
DIMENSION = 8  # 8x8 board
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 60  # For animations
IMAGES = {}
PIECES_TO_IMAGES = {
    'P': 'p',  # White Pawn
    'R': 'r',  # White Rook
    'N': 'n',  # White Knight
    'B': 'b',  # White Bishop
    'Q': 'q',  # White Queen
    'K': 'k',  # White King
    'p': 'bp',  # Black Pawn
    'r': 'br',  # Black Rook
    'n': 'bn',  # Black Knight
    'b': 'bb',  # Black Bishop
    'q': 'bq',  # Black Queen
    'k': 'bk'   # Black King
}
WHITE = True  # White starts first
AI_THINKING = False  # Flag to indicate if AI is processing

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True

# Draw the chessboard
def draw_board(screen):
    colors = [pygame.Color("white"), pygame.Color("gray")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[(r + c) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

# Load images
def load_images():
    pieces = ['bp', 'bk', 'bn', 'bb', 'br', 'bq',
              'b', 'k', 'n', 'p', 'q', 'r']
    for piece in pieces:
        IMAGES[piece] = pygame.transform.scale(
            pygame.image.load(f"images/{piece}.png"), (SQ_SIZE, SQ_SIZE))

# Draw the pieces on the board
def draw_pieces(screen, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board.piece_at((DIMENSION-1-r) * DIMENSION + c)
            if piece:
                piece_image = IMAGES[PIECES_TO_IMAGES[piece.symbol()]]
                screen.blit(piece_image, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))


def ai_move(s, model):
    chess_moves = {}
    print(s.board.legal_moves)
    for move in s.board.legal_moves:
        s.board.push(move)
        b = s.board_to_matrix()
        input_tensor = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        chess_moves[move] = output.item()
        s.board.pop()

    
    if not chess_moves:
        print("No legal moves available for AI.")
        return

    best_move = max(chess_moves, key=chess_moves.get)
    s.board.push(best_move)
    print(f"AI plays: {best_move}")

    return s
    


def load_model(path):
    try:
        vals = torch.load(path, weights_only=True)
        model = ChessModel()
        model.load_state_dict(vals)
        return model
    except Exception as e:
        print(f"Error loading AI model: {e}")
        sys.exit()

def main():

    model = load_model("nets/value.pth") # load ai model
    load_images() # load the images for the gui
    s = State()
    running = True
    selected_square = None
    AI_THINKING = True


    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not AI_THINKING:
                location = pygame.mouse.get_pos()  # (x, y)
                col = location[0] // SQ_SIZE
                row = location[1] // SQ_SIZE
                square = chess.square(col, 7 - row)  # Convert to chess square
                if selected_square is None:
                    piece = s.board.piece_at(square)
                    if piece and piece.color == chess.BLACK:
                        selected_square = square
                else:
                    move = chess.Move(selected_square, square)
                    
                    if move in s.board.legal_moves:
                        s.board.push(move)
                        selected_square = None
                        AI_THINKING = True
                    else:
                        print("Invalid move")
                        selected_square = None
            elif AI_THINKING:
                s = ai_move(s, model) # update state
                AI_THINKING = False
                
            
        draw_board(screen)
        draw_pieces(screen, s.board)

        if s.board.is_game_over():
            print("Game Over")

        pygame.display.flip()
        clock.tick(60)  # limits FPS to 60

    pygame.quit()

if __name__ == "__main__":
    main()