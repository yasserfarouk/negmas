#!/usr/bin/env bash

# Ensure all docs dependencies are installed
uv sync --all-extras --dev --quiet

./update.sh
make html
mkdir -p "./_build/html/figs"
mkdir -p "./_build/html/tutorials/figs"
[ -d "./figs" ] && cp ./figs/*.* "./_build/html/figs/"
[ -d "./figs" ] && cp ./figs/*.* "./_build/html/tutorials/figs"
[ -d "../notebooks/figs" ] && cp ../notebooks/figs/*.* "./_build/html/figs/"
[ -d "../notebooks/figs" ] && cp ../notebooks/figs/*.* "./_build/html/tutorials/figs"
# open _build/html/index.html
echo "done!!"
