#!/bin/bash
nice -n 10 cutechess-cli -engine cmd=~/personal/viridithas/target/release/viridithas -engine cmd=$1 -each timemargin=400 proto=uci tc=8+0.08 option.Hash=16 -rounds 2500 -concurrency 12 -openings file=book.epd order=random format=epd -repeat -games 2 -pgnout stctests.pgn -event STC_REGRESSION_TEST
echo "Test done at 8s+0.08s"
