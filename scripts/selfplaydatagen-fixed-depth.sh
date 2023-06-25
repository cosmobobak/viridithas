#!/bin/bash
nice -n 10 ~/cutechess-cli-folder/cutechess-cli \
    -engine cmd=target/release/viridithas \
    -engine cmd=target/release/viridithas \
    -each proto=uci depth=8 tc=120+12 -rounds 320000 -concurrency 60 -openings file=../uhobook.pgn format=pgn -pgnout selfplay.pgn fi > selfplaylog.txt
