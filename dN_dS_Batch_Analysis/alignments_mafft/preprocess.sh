#!/bin/bash

python rename_files.py ./

cd ../scripts
chmod 775 *
chmod 775 pal2nal_v14/*
# Now, move to the `raw_data` directory
cd ../alignments_mafft
# From this directory, run the following commands
for i in *.fasta
do
name=$( echo $i | sed 's/\.fasta//' )
name2=$( echo $name | sed 's/\_raw//' )
../scripts/one_line_fasta.pl $i 
mv $name"_one_line.fa" $name2".fasta"
done
