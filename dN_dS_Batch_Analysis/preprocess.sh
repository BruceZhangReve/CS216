#!/bin/bash

# First, give permissions to the `scripts` 
# directory so that you can execute the
# various scripts saved in this directory
# throughout this tutorial
# Once you are in the `scripts` directory,
# run the following two commands
cd scripts
chmod 775 *
chmod 775 pal2nal_v14/*
# Now, move to the `raw_data` directory
cd ../raw_data
# From this directory, run the following commands
for i in *raw.fasta
do
name=$( echo $i | sed 's/\.fasta//' )
name2=$( echo $name | sed 's/\_raw//' )
../scripts/one_line_fasta.pl $i 
mv $name"_one_line.fa" $name2".fasta"
done

for i in *init.fasta
do
name=$( echo $i | sed 's/\.fasta//' )
name2=$( echo $name | sed 's/\_raw//' )
../scripts/one_line_fasta.pl $i 
mv $name"_one_line.fa" $name2".fasta"
done