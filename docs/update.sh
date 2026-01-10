#!/usr/bin/env bash
#
# IMPORTANT: This script must be run before pushing to ensure documentation
# notebooks are properly converted to RST format.
#
# Usage: cd docs && ./update.sh
#
# This script:
# 1. Converts Jupyter notebooks to RST format for Sphinx documentation
# 2. Copies notebook files and images to the docs directory
# 3. Adds download links to tutorial pages
#
# Run this script whenever notebooks in notebooks/ or notebooks/tutorials/ are modified.
#

set -e  # Exit on error

echo "Converting notebooks to RST format..."

rm -f ./tutorials.rst
cp ./tutorials_template ./tutorials.rst
echo "    " >> ./tutorials.rst

echo "Converting overview.ipynb..."
jupyter nbconvert --TagRemovePreprocessor.remove_cell_tags "remove_cell" --to rst ../notebooks/overview.ipynb

echo "Converting getting_started.ipynb..."
jupyter nbconvert --TagRemovePreprocessor.remove_cell_tags "remove_cell" --to rst ../notebooks/getting_started.ipynb

echo "Converting tutorial notebooks..."
for notebook in $(find ../notebooks/tutorials -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" -not -path "*/.ipynb_checkpoints/*" | sort -g) ; do
    echo "  Converting $notebook..."
    jupyter nbconvert --TagRemovePreprocessor.remove_cell_tags "remove_cell" --to rst "$notebook"
    jupyter nbconvert --TagRemovePreprocessor.remove_cell_tags "remove_cell" --to rst "$notebook"
    jupyter nbconvert --TagRemovePreprocessor.remove_cell_tags "remove_cell" --to rst "$notebook"
    filename1=${notebook##*/}
    filename=${filename1%??????}
    echo "    tutorials/$filename" >> ./tutorials.rst
done

echo "Organizing output files..."
rm -rf ./tutorials
mkdir ./tutorials
mv ../notebooks/tutorials/*.rst ./tutorials
for fils in ../notebooks/tutorials/*_files ; do
    if [ -d "$fils" ]; then
        mv "$fils" ./tutorials
    fi
done
mv ../notebooks/overview.rst .
mv ../notebooks/getting_started.rst .
for fils in ../notebooks/*_files ; do
    if [ -d "$fils" ]; then
        mv "$fils" .
    fi
done

echo "Copying figures..."
mkdir -p ./figs
for ext in png jpeg jpg pdf; do
    cp -f ../notebooks/figs/*.$ext ./figs 2>/dev/null || true
done
mkdir -p ./tutorials/figs
for ext in png jpeg jpg pdf; do
    cp -f ../notebooks/tutorials/figs/*.$ext ./tutorials/figs 2>/dev/null || true
done

echo "Copying notebooks for download..."
mkdir -p ./tutorials/notebooks
for f in ../notebooks/tutorials/*.ipynb ; do
    cp "$f" ./tutorials/notebooks
done

for f in ../notebooks/tutorials/*.gif ; do
    if [ -f "$f" ]; then
        cp "$f" ./tutorials/notebooks
    fi
done

echo "Adding download links to RST files..."
for rstfile in ./tutorials/*.rst; do
    filename1=${rstfile##*/}
    filename=${filename1%????}
    echo "" >> "$rstfile"
    echo "" >> "$rstfile"
    echo "Download :download:\`Notebook<notebooks/$filename.ipynb>\`." >> "$rstfile"
done

echo "------------------------------------------------"
echo "Cleaning previous build..."
make clean
rm -rf api

echo "Documentation notebooks have been updated successfully!"
echo "You can now build the documentation with: make html"
