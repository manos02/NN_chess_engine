import pygame
from state import State
import chess
from play import alphaBetaMax, load_model, human_move


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


def main():

    model = load_model("nets/value.pth") # load ai model
    load_images() # load the images for the gui
    s = State()
    running = True
    selected_square = None
    ai_thinking = True


    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not ai_thinking:
                location = pygame.mouse.get_pos()  # (x, y)
                col = location[0] // SQ_SIZE
                row = location[1] // SQ_SIZE
                square = chess.square(col, 7 - row)  # Convert to chess square
                s, selected_square, ai_thinking = human_move(selected_square, square, s, ai_thinking)
            elif ai_thinking:
                best_score, best_move = alphaBetaMax(3, s, -float("inf"), float("inf"), True, model) # update state
                print(f"AI MOVE: {best_move}")
                s.board.push(best_move)
                ai_thinking = False
            
        draw_board(screen)
        draw_pieces(screen, s.board)

        if s.board.is_game_over():
            print("Game Over")

        pygame.display.flip()
        clock.tick(60)  # limits FPS to 60

    pygame.quit()

if __name__ == "__main__":
    main()