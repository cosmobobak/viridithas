#!/bin/bash
nice -n 10 cutechess-cli -engine cmd=$1 name=$2 -engine cmd=$3 name=$4 -each timemargin=400 proto=uci tc=8+0.08 -rounds 2500 -concurrency 60 -openings file=../uhobook.pgn order=random format=pgn -repeat -games 2 -pgnout stctests.pgn -event MATCHUP
echo "Test done at 8s+0.08s"
