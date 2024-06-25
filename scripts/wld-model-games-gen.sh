#!/bin/bash
nice -n 10 cutechess-cli -engine cmd=target/release/viridithas -engine cmd=target/release/viridithas -each timemargin=400 proto=uci tc=16+0.16 -rounds 20000 -concurrency 64 -openings file=../uhobook.pgn order=random format=pgn -pgnout wld-model-games.pgn -recover
