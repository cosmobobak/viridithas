#!/bin/bash
nice -n 10 cutechess-cli -engine cmd=target/release/viridithas name=$1 -engine cmd=dev name=dev -each timemargin=400 proto=uci tc=60+0.6 -rounds 2500 -concurrency 60 -openings file=../uhobook.pgn order=random format=pgn -repeat -games 2 -pgnout ltctests.pgn -event LTC_REGRESSION_TEST
echo "Test done at 1m+0.6s"
