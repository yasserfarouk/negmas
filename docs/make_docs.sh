#!/usr/bin/env bash
./update.sh
make html
mkdir -p "./_build/html/figs"
cp "../notebooks/figs/*.*" "./_build/html/figs/"
cp "../notebooks/figs/*.*" "./_build/html/tutorials/figs"
# open _build/html/index.html
