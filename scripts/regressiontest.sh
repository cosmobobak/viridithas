#!/bin/bash
nice -n 10 cutechess-cli -engine cmd=target/release/viridithas name=$1 -engine cmd=dev name=dev -each timemargin=400 proto=uci tc=100/8+0.08 -rounds 2500 -concurrency 60 -openings file=../uhobook.pgn format=pgn -repeat -games 2
