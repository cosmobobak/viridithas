#!/bin/bash
echo "Processing $1" >> texelconversionlog.txt
nice -n 10 pypy scripts/texelconversion.py $1 $2 >> texelconversionlog.txt
