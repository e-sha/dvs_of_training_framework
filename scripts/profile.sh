#!/bin/bash
pstat_file=$(mktemp)
img_file=$(mktemp).png

python3 -m cProfile -o ${pstat_file} "$@"
gprof2dot -f pstats ${pstat_file} | dot -Tpng -o ${img_file} && feh ${img_file}
rm $pstat_file $img_file
