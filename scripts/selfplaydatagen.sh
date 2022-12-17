#!/bin/bash
nice -n 10 ~/cutechess-cli-folder/cutechess-cli -engine cmd=target/release/viridithas -engine cmd=target/release/viridithas -each timemargin=400 proto=uci tc=8+0.08 -rounds 320000 -concurrency 60 -openings file=../uhobook.pgn format=pgn -pgnout selfplay.pgn fi > selfplaylog.txt
