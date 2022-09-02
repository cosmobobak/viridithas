
import chess
import chess.pgn
import tqdm

PGN = "selfplay.pgn" # 366,392 games

def main():
    counter = 0
    pbar = tqdm.tqdm()
    with open(f"{PGN}", "r") as pgn:
        while True:
            if counter & 0xFF == 0:
                pbar.update(0xFF)
            game = chess.pgn.read_game(pgn)
            if game is None: # EOF
                break
            counter += 1
    pbar.close()
    print(f"Found {counter} games.")

main()
