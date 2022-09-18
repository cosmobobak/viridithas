#!/bin/bash
nice -n 10 target/release/viridithas --tune $1 --examples 25000000 --resume > optimisation_log.txt
