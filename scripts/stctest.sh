#!/bin/bash
nice -n 10 cutechess-cli -engine cmd=target/release/viridithas stderr=regteststderr.txt name=$1 -engine cmd=dev name=dev -each timemargin=400 proto=uci tc=100/8+0.08 -rounds 2500 -concurrency 60 -openings file=../uhobook.pgn order=random format=pgn -repeat -games 2 -pgnout stctests.pgn -event STC_REGRESSION_TEST
echo "Test done at 8s+0.08s"
