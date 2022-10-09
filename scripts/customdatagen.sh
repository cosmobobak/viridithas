#!/bin/bash
nice -n 10 ~/cutechess-cli-folder/cutechess-cli -engine cmd=$1 name=$2 -engine cmd=$3 name=$4 -each timemargin=400 proto=uci tc=100/8+0.08 -rounds 256000 -concurrency 60 -openings file=../uhobook.pgn format=pgn -recover -pgnout selfplay.pgn fi > selfplaylog.txt
