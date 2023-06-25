#!/bin/bash
nice -n 10 cutechess-cli -engine cmd=target/release/viridithas stderr=regteststderr.txt -engine cmd=$1 -each timemargin=400 proto=uci tc=60+0.6 option.Hash=256 -rounds 500 -concurrency 60 -openings file=../uhobook.pgn order=random format=pgn -repeat -games 2 -pgnout ltctest_${2}.pgn -event STC_REGRESSION_TEST
echo "Test done at 60s+0.6s"
