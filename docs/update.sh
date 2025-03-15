#!/usr/bin/env bash
rm ./tutorials.rst
cp ./tutorials_template ./tutorials.rst
echo "    " >> ./tutorials.rst
jupyter nbconvert --TagRemovePreprocessor.remove_cell_tags "remove_cell" --to rst ../notebooks/overview.ipynb
jupyter nbconvert --TagRemovePreprocessor.remove_cell_tags "remove_cell" --to rst ../notebooks/getting_started.ipynb
for notebook in `ls ../notebooks/tutorials/*.ipynb | sort -g` ; do
    jupyter nbconvert --TagRemovePreprocessor.remove_cell_tags "remove_cell" --to rst "$notebook"
    jupyter nbconvert --TagRemovePreprocessor.remove_cell_tags "remove_cell" --to rst "$notebook"
    jupyter nbconvert --TagRemovePreprocessor.remove_cell_tags "remove_cell" --to rst "$notebook"
    filename1=${notebook##*/}
    filename=${filename1%??????}
    echo "    tutorials/$filename" >> ./tutorials.rst
done
rm -r ./tutorials
mkdir ./tutorials
mv ../notebooks/tutorials/*.rst ./tutorials
for fils in ../notebooks/tutorials/*_files ; do
    mv $fils ./tutorials
done
mv ../notebooks/overview.rst .
mv ../notebooks/getting_started.rst .
mkdir ./figs
for ext in png jpg pdf; do
    cp ../notebooks/figs/*.$ext ./figs

    mkdir ./tutorials/notebooks
    for f in ../notebooks/tutorials/*.ipynb ; do
        cp $f ./tutorials/notebooks
    done

    for f in ../notebooks/tutorials/*.gif ; do
        cp $f ./tutorials/notebooks
    done
    for rstfile in ./tutorials/*.rst; do
        filename1=${rstfile##*/}
        filename=${filename1%????}
        echo "" >> $rstfile
        # echo ".. only:: builder_html">> $rstfile
        echo "" >> $rstfile
        echo "Download :download:\`Notebook<notebooks/$filename.ipynb>\`." >> $rstfile
    done
done
echo "------------------------------------------------"
make clean
rm -r api
