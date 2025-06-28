#!/bin/bash
./cutechess-cli \
  -engine name=$1 \
    cmd=dev \
  -engine name=master \
    cmd=target/release/viridithas \
  -games 2 \
  -rounds 50000 \
  -sprt elo0=0.0 elo1=3.0 alpha=0.05 beta=0.05 \
  -each proto=uci tc=40+0.4 \
  -openings order=random file=./UHO_Lichess_4852_v1.epd format=epd \
  -concurrency 32 \
  -ratinginterval 30
