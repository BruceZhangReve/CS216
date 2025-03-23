#Usage: bash build_tree.sh "CoronavirusNL63" "/u02/ligezhang/drug_design/RAxML_mafft" "/u02/ligezhang/drug_design/alignments_mafft"

#!/bin/bash
virus_name=$1
output_mafft_dir=$2
alignment_dir=$3
#virus_name="CoronavirusNL63"
#output_mafft_dir="/u02/ligezhang/drug_design/RAxML_mafft"

for i in ${alignment_dir}/${virus_name}*.phy; do
    full_name=$(basename "$i")
    Name=${full_name%%_nuc*}

    raxmlHPC -m GTRGAMMA -f d -p 12345 -s "$i" -n ${Name}_mafft -w ${output_mafft_dir}

    if [ $? -ne 0 ]; then
        echo "raxmlHPC failed for $i, skipping..."
        continue
    fi
done


#bash build_tree.sh Poxviridae /u02/ligezhang/monkey_pox/RAxML_mafft /u02/ligezhang/monkey_pox/alignments_mafft