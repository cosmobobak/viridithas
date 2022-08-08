#!/bin/bash
nice -n 10 ~/cutechess-cli-folder/cutechess-cli -engine cmd=target/release/viridithas name=dev -engine cmd=target/release/viridithas name=dev2 -each timemargin=400 proto=uci tc=100/8+0.08 -rounds 300000 -concurrency 60 -openings file=../uhobook.pgn format=pgn -pgnout selfplay.pgn fi > /dev/null
