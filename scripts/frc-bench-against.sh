#!/bin/bash
nice -n 10 cutechess-cli -engine cmd=target/release/viridithas name="Viridithas" -engine cmd=$1 -each timemargin=400 proto=uci tc=8+0.08 option.Hash=16 -rounds 2500 -concurrency 60 -openings file=../dfrc.epd order=random format=epd -repeat -games 2 -pgnout stctests.pgn -event STC_REGRESSION_TEST -variant fischerandom
echo "Test done at 8s+0.08s"
