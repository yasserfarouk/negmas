#!/usr/bin/env bash
rm ./tutorials.rst
cp ./tutorials_template ./tutorials.rst
jupyter nbconvert --to rst ../notebooks/overview.ipynb
for notebook in `ls ../notebooks/tutorials/*.ipynb | sort -g` ; do
    jupyter nbconvert --to rst $notebook
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
mkdir ./tutorials/notebooks
for notebook in ../notebooks/tutorials/*.ipynb ; do
    cp $notebook ./tutorials/notebooks
done

for rstfile in ./tutorials/*.rst; do
    filename1=${rstfile##*/}
    filename=${filename1%????}
    echo "" >> $rstfile
    echo ".. only:: builder_html">> $rstfile
    echo "" >> $rstfile
    echo "    Download :download:\`Notebook<notebooks/$filename.ipynb>\`." >> $rstfile
    echo "" >> $rstfile
    echo "" >> $rstfile
done
echo "------------------------------------------------"
make clean
rm -r api
make latexpdf

#echo "" >> ./readme.rst
#echo ".. only:: builder_html">> ./readme.rst
#echo "" >> ./readme.rst
#echo "    Download PDF version :download:\`here<_build/latex/negmas.pdf>\`." >> ./readme.rst
#echo "" >> ./readme.rst
#echo "" >> ./readme.rst

make html
open _build/html/index.html
open _build/latex/negmas.pdf


