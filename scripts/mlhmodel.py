import chess
import chess.pgn
from sklearn.linear_model import LinearRegression

# read command line args
import sys
if len(sys.argv) != 2:
    print("Usage: mlhmodel.py <INPUTPGN>")
    exit(1)
PGN = sys.argv[1]

# list of (phase, move) tuples
xs: "list[tuple[int, int]]" = []
# list of moves-left values
ys: "list[int]" = []

pawn_phase, knight_phase, bishop_phase, rook_phase, queen_phase = [1, 10, 10, 20, 40]

def main():
    game_counter = 0
    print("Reading games...")
    with open(PGN, "r") as pgn:
        while True:
            pos_counter = 0
            game = chess.pgn.read_game(pgn)
            if game is None: # EOF
                break
            game_counter += 1
            nodes = game.mainline()
            for node in nodes:
                board = node.board()
                phase = 0
                phase += chess.popcount(board.pawns) * pawn_phase
                phase += chess.popcount(board.knights) * knight_phase
                phase += chess.popcount(board.bishops) * bishop_phase
                phase += chess.popcount(board.rooks) * rook_phase
                phase += chess.popcount(board.queens) * queen_phase
                xs.append((phase, board.ply()))
                pos_counter += 1
            for i in range(pos_counter):
                ys.append(pos_counter - i)
        assert len(xs) == len(ys), f"{len(xs) = }, {len(ys) = }"
    print(f"Read {game_counter} games, {len(xs)} positions")
    print("Training LR model...")
    # fit model
    model = LinearRegression()
    model.fit(xs, ys)
    # write model
    weights = model.coef_
    intercept = model.intercept_
    print(f"{weights = }")
    print(f"{intercept = }")

main()