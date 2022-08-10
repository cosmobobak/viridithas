import sys
import chess
import chess.pgn
from tqdm import tqdm

PGN = "lichess_elite_2022-03.pgn" # 366,392 games
N_GAMES = 366392
MAX_GAMES = 200_000

def main():
    counter = 0
    wins, draws, losses = 0, 0, 0
    endpoint = min(N_GAMES, MAX_GAMES)
    with open(f"../{PGN}", "r") as pgn:
        with open("../texel_data.txt", "w") as texel_data:
            while counter < MAX_GAMES:
                if counter & 0xFF == 0:
                    print(f"Processed {counter} games ({counter / endpoint * 100.0:.2f}% done)        \r", end="")
                    # flush stdout
                    sys.stdout.flush()
                game = chess.pgn.read_game(pgn)
                if game is None: # EOF
                    break
                result = game.headers["Result"]
                if result == "1-0":
                    result = 1.0
                    wins += 1
                elif result == "0-1":
                    result = 0.0
                    losses += 1
                elif result == "1/2-1/2":
                    result = 0.5
                    draws += 1
                else:
                    print(f"Unknown result: {result}")
                    exit(1)
                moves = game.mainline_moves()
                board = chess.Board()
                for i, move in enumerate(moves):
                    if i >= 5: # only use moves after 5th
                        texel_data.write(f"{board.fen()} {result}\n")
                    board.push(move)
                counter += 1
    print(f"Processed {counter} games ({counter / endpoint * 100.0:.2f}% done)")
    print(f"Wins:   {wins}")
    print(f"Draws:  {draws}")
    print(f"Losses: {losses}")

main()