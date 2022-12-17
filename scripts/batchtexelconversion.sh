#!/bin/bash
for i in $1/*.pgn
do
    echo "Processing $i" >> texelconversionlog.txt
    nice -n 10 pypy scripts/texelconversion.py $i $2 >> texelconversionlog.txt
done