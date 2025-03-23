#!/bin/bash
#virus_name="Adenoviridae"
#alignments_mafft_dir="/u02/ligezhang/drug_design/alignments_mafft"
#output_mafft_dir="/u02/ligezhang/drug_design/RAxML_mafft"

#Example:
#bash modify_tree.sh "Adenoviridae" "/u02/ligezhang/drug_design/alignments_mafft" "/u02/ligezhang/drug_design/RAxML_mafft"
virus_name=$1
alignments_mafft_dir=$2
output_mafft_dir=$3

# Function to process files
process_file() {
    local dir=$1
    local input_tree=$2
    local output_tree=$3
    
    echo "Parsing tree in directory $dir"
    
    if [[ -f $input_tree ]]; then
        # Count the number of branches (internal nodes and leaves) from the tree file
        num_branches=$(( $(grep -o ',' "$input_tree" | wc -l) + 1 ))
        
        # Write the header line to the output tree file
        printf "%s  1\n" "$num_branches" > "$output_tree"
        
        # Remove branch lengths and append the cleaned tree to the output file
        sed 's/:[0-9]*\.[0-9]*//g' "$input_tree" >> "$output_tree"
        echo "Processed and saved to $output_tree"
    else
        echo "Tree file $input_tree not found in directory $dir"
    fi
}

# Find and process the files
files=$(find "$output_mafft_dir" -type f -name "RAxML_bestTree*${virus_name}*mafft")

for file in $files; do
    echo "Processing file: $file"
    dir=$(dirname "$file")
    output_tree="${file}_nobl.tree"
    process_file "$dir" "$file" "$output_tree"
done

