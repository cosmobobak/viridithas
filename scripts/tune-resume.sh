#!/bin/bash
nice -n 10 target/release/viridithas --tune --examples 25000000 --resume --params /afs/inf.ed.ac.uk/user/s20/s2079150/virtue/params/localsearch507.txt > optimisation_log.txt
