#!/usr/bin/env bash
jupyter nbconvert --to rst ../notebooks/overview.ipynb
mv ../notebooks/overview.rst .
make clean
rm -r api
make html
make latexpdf
open _build/html/index.html
open _build/latex/negmas.pdf


