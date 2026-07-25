import threading

import pygame
from state import State
import chess
from play import alphaBetaMax, load_model, human_move

# pygame setup
pygame.init()

# Constants
BOARD_SIZE = 800
PANEL_WIDTH = 280  # side panel with captures, score and AI status
WIDTH, HEIGHT = BOARD_SIZE + PANEL_WIDTH, BOARD_SIZE
DIMENSION = 8  # 8x8 board
SQ_SIZE = BOARD_SIZE // DIMENSION
SMALL_SQ = 40  # size of the captured piece icons
MAX_FPS = 60  # For animations
IMAGES = {}
SMALL_IMAGES = {}
START_COUNTS = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1}
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

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Convert a chess square to (col, row) on screen, given the board orientation
def square_to_screen(square, flipped):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    if flipped:
        return 7 - file, rank
    return file, 7 - rank

# Convert a screen (col, row) back to a chess square
def screen_to_square(col, row, flipped):
    if flipped:
        return chess.square(7 - col, row)
    return chess.square(col, 7 - row)

# Draw the chessboard
def draw_board(screen, selected_square, last_move, flipped):

    last_move_color_from = pygame.Color("blue")    # Color for source square
    last_move_color_to = pygame.Color("red")

    colors = [pygame.Color("white"), pygame.Color("gray")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[(r + c) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

    if selected_square is not None:
            col, row = square_to_screen(selected_square, flipped)
            pygame.draw.rect(screen, last_move_color_from, pygame.Rect(col*SQ_SIZE, row*SQ_SIZE, SQ_SIZE, SQ_SIZE), 4)

    # Highlight the last move's source and destination squares
    if last_move is not None:
        # Highlight source square
        col_from, row_from = square_to_screen(last_move.from_square, flipped)
        pygame.draw.rect(screen, last_move_color_from, pygame.Rect(col_from*SQ_SIZE, row_from*SQ_SIZE, SQ_SIZE, SQ_SIZE), 4)

        # Highlight destination square
        col_to, row_to = square_to_screen(last_move.to_square, flipped)
        pygame.draw.rect(screen, last_move_color_to, pygame.Rect(col_to*SQ_SIZE, row_to*SQ_SIZE, SQ_SIZE, SQ_SIZE), 4)

# Load images
def load_images():
    pieces = ['bp', 'bk', 'bn', 'bb', 'br', 'bq',
              'b', 'k', 'n', 'p', 'q', 'r']
    for piece in pieces:
        image = pygame.image.load(f"images/{piece}.png")
        IMAGES[piece] = pygame.transform.scale(image, (SQ_SIZE, SQ_SIZE))
        SMALL_IMAGES[piece] = pygame.transform.scale(image, (SMALL_SQ, SMALL_SQ))

# Draw the pieces on the board
def draw_pieces(screen, board, flipped):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            c, r = square_to_screen(square, flipped)
            piece_image = IMAGES[PIECES_TO_IMAGES[piece.symbol()]]
            screen.blit(piece_image, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))


# Draw a centered text label, returns its rect
def draw_text(screen, text, size, center, color=pygame.Color("white")):
    font = pygame.font.SysFont("Arial", size, bold=True)
    surface = font.render(text, True, color)
    rect = surface.get_rect(center=center)
    screen.blit(surface, rect)
    return rect

# Draw a button, returns its rect
def draw_button(screen, text, center, size=(220, 70)):
    rect = pygame.Rect(0, 0, *size)
    rect.center = center
    pygame.draw.rect(screen, pygame.Color("gray20"), rect)
    pygame.draw.rect(screen, pygame.Color("white"), rect, 3)
    draw_text(screen, text, 30, rect.center)
    return rect

# Draw a left-aligned text label, returns its rect
def draw_text_left(screen, text, size, topleft, color=pygame.Color("white")):
    font = pygame.font.SysFont("Arial", size, bold=True)
    surface = font.render(text, True, color)
    rect = surface.get_rect(topleft=topleft)
    screen.blit(surface, rect)
    return rect

# Pieces of `color` that are missing from the board, best first
def captured_pieces(board, color):
    captured = []
    for piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
        missing = START_COUNTS[piece_type] - len(board.pieces(piece_type, color))
        captured += [chess.Piece(piece_type, color).symbol()] * max(missing, 0)
    return captured

# Draw a row of captured piece icons, returns the y below them
def draw_captures(screen, captured, x, y):
    if not captured:
        draw_text_left(screen, "-", 22, (x, y), pygame.Color("gray60"))
        return y + SMALL_SQ
    per_row = (PANEL_WIDTH - 40) // SMALL_SQ
    for i, symbol in enumerate(captured):
        col, row = i % per_row, i // per_row
        screen.blit(SMALL_IMAGES[PIECES_TO_IMAGES[symbol]], (x + col*SMALL_SQ, y + row*SMALL_SQ))
    return y + ((len(captured) - 1) // per_row + 1) * SMALL_SQ

# Side panel: match score, captured pieces and AI status
def draw_panel(screen, board, human_color, score, thinking):
    x = BOARD_SIZE + 20
    pygame.draw.rect(screen, pygame.Color("gray10"), pygame.Rect(BOARD_SIZE, 0, PANEL_WIDTH, HEIGHT))

    side = "White" if human_color == chess.WHITE else "Black"
    draw_text_left(screen, f"You play {side}", 26, (x, 20))

    draw_text_left(screen, "Score", 24, (x, 80), pygame.Color("gray70"))
    draw_text_left(screen, f"You {score['human']} - {score['ai']} AI", 34, (x, 110))
    if score["draws"]:
        draw_text_left(screen, f"Draws: {score['draws']}", 22, (x, 155), pygame.Color("gray70"))

    ai_color = not human_color
    draw_text_left(screen, "You captured", 24, (x, 210), pygame.Color("gray70"))
    y = draw_captures(screen, captured_pieces(board, ai_color), x, 245)

    draw_text_left(screen, "AI captured", 24, (x, y + 40), pygame.Color("gray70"))
    draw_captures(screen, captured_pieces(board, human_color), x, y + 75)

    if thinking:
        draw_text_left(screen, "AI thinking...", 26, (x, HEIGHT - 60), pygame.Color("gold"))

# Start screen: pick a side. Returns chess.WHITE / chess.BLACK, or None if the window was closed.
def choose_color(screen):
    while True:
        screen.fill(pygame.Color("gray10"))
        draw_text(screen, "Choose your side", 50, (WIDTH//2, HEIGHT//3))
        white_rect = draw_button(screen, "White", (WIDTH//2 - 140, HEIGHT//2))
        black_rect = draw_button(screen, "Black", (WIDTH//2 + 140, HEIGHT//2))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if white_rect.collidepoint(event.pos):
                    return chess.WHITE
                elif black_rect.collidepoint(event.pos):
                    return chess.BLACK
        clock.tick(MAX_FPS)

# Text describing how the game ended
def result_text(board):
    outcome = board.outcome()
    if outcome is None:
        return "Game over"
    if outcome.winner is None:
        return f"Draw ({outcome.termination.name.replace('_', ' ').title()})"
    winner = "White" if outcome.winner == chess.WHITE else "Black"
    return f"{winner} wins by {outcome.termination.name.replace('_', ' ').lower()}"

# Game over popup. Returns True to play again, False to quit.
def game_over_popup(screen, board, draw_game):
    while True:
        draw_game()
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        screen.blit(overlay, (0, 0))

        draw_text(screen, "Game Over", 60, (WIDTH//2, HEIGHT//3 - 40))
        draw_text(screen, result_text(board), 32, (WIDTH//2, HEIGHT//3 + 30))
        draw_text(screen, f"Result: {board.result()}", 28, (WIDTH//2, HEIGHT//3 + 80))
        again_rect = draw_button(screen, "Play Again", (WIDTH//2 - 140, HEIGHT//2 + 100))
        quit_rect = draw_button(screen, "Quit", (WIDTH//2 + 140, HEIGHT//2 + 100))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if again_rect.collidepoint(event.pos):
                    return True
                elif quit_rect.collidepoint(event.pos):
                    return False
        clock.tick(MAX_FPS)


# Run the search in a background thread on a copy of the board, so the window stays responsive
def start_search(model, board, ai_is_white, result):
    def run():
        result.append(alphaBetaMax(3, State(board.copy()), -float("inf"), float("inf"), ai_is_white, model))
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


# Play a single game. Returns True to play again, False to quit.
def play_game(model, human_color, score):
    s = State()
    flipped = (human_color == chess.BLACK)  # keep the human's pieces at the bottom
    ai_is_white = (human_color == chess.BLACK)  # the AI maximizes when it plays white
    ai_thinking = (s.board.turn != human_color)
    ai_thread = None
    ai_result = []
    selected_square = None
    last_move = None

    def draw_game():
        draw_board(screen, selected_square, last_move, flipped)
        draw_pieces(screen, s.board, flipped)
        draw_panel(screen, s.board, human_color, score, ai_thinking)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN and not ai_thinking and event.pos[0] < BOARD_SIZE:
                col = event.pos[0] // SQ_SIZE
                row = event.pos[1] // SQ_SIZE
                square = screen_to_square(col, row, flipped)
                s, selected_square, ai_thinking, move = human_move(selected_square, square, s, human_color)
                if move is not None:
                    last_move = move

        if ai_thinking and ai_thread is None and not s.board.is_game_over():
            ai_result = []
            ai_thread = start_search(model, s.board, ai_is_white, ai_result)

        if ai_thread is not None and not ai_thread.is_alive():
            best_score, best_move = ai_result[0]
            print(f"AI MOVE: {best_move}")
            s.board.push(best_move)
            last_move = best_move
            ai_thinking = False
            ai_thread = None

        draw_game()

        if s.board.is_game_over():
            print("GAME OVER")
            print("RESULT: ", s.board.result())
            update_score(s.board, human_color, score)
            return game_over_popup(screen, s.board, draw_game)

        pygame.display.flip()
        clock.tick(MAX_FPS)


# Add the finished game to the running match score
def update_score(board, human_color, score):
    winner = board.outcome().winner if board.outcome() else None
    if winner is None:
        score["draws"] += 1
    elif winner == human_color:
        score["human"] += 1
    else:
        score["ai"] += 1


def main():
    model = load_model("nets/value.pth") # load ai model
    load_images() # load the images for the gui
    score = {"human": 0, "ai": 0, "draws": 0}

    while True:
        human_color = choose_color(screen)
        if human_color is None:
            break
        if not play_game(model, human_color, score):
            break

    pygame.quit()

if __name__ == "__main__":
    main()
