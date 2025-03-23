#!/bin/bash
# Run from this directory where
# this README.md file is
# Note: you need to change name respectively!
virus_name=$1
#virus_name="Adenoviridae"

# Change directory
current_dir=$(pwd)
cd "./alignments_mafft" || exit

# Loop through files matching virus name prefix
for i in ${virus_name}*_nuc_mafft_aln.fasta; do
    # Calculate the number of sequences in the FASTA file
    num=$(grep '>' "$i" | wc -l)
    
    # Get the length of the first sequence (assuming all sequences are the same length)
    len=$(awk 'NR==2{print length}' "$i")
    
    # Execute Perl script with necessary parameters
    perl ../scripts/FASTAtoPHYL.pl "$i" $num $len
    
    # Rename output file from .phy to _mafft.phy
    mv "${i%.fasta}.phy" "${i%.fasta}_mafft.phy"
done

# Change directory back
cd "$current_dir"


