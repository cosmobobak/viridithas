#!/bin/bash
mv nnue temp-nnue
cargo rustc --release -- -C target-cpu=native 2> /dev/null
mv ../nets/$1 nnue
cargo rustc --release -- -C target-cpu=native
mv nnue ../nets/$1
mv temp-nnue nnue
mv target/release/viridithas ../viridithas-dev-$1
