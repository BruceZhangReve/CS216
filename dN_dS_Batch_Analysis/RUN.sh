#!/bin/bash
virus="Coronaviridae" #The family name of the virus

#Note: use direct path to avoid issues
path_to_alignments="/data/lige/genome/dN_dS_Batch_Analysis/alignments_mafft" 
path_to_trees="/data/lige/genome/dN_dS_Batch_Analysis/RAxML_mafft" 
template_ctl_path="/data/lige/genome/dN_dS_Batch_Analysis/paml/codeml.ctl" 
output_dir="/data/lige/genome/dN_dS_Batch_Analysis/result" 

# Check if there are any files that match the pattern in the RAxML_mafft directory
[[ "${path_to_trees}" != */ ]] && path_to_trees="${path_to_trees}/"
if ls ${path_to_trees}*"${virus}"* 1> /dev/null 2>&1; then
    echo "Existing files found for ${virus}, deleting..."
    #If tree files exists, delete them, otherwise will have a problem
    rm ${path_to_trees}*"${virus}"*
else
    echo "No existing files to delete for ${virus}."
fi


#Make a gene: 1 line of header + 1 line of senquence
current_dir=$(pwd)
cd "$path_to_alignments"
if ! bash preprocess.sh; then
    echo "Preprocess(structure tune) failed."
    # Return to the original directory before exiting
    cd "$current_dir"
    exit 1
fi
echo "Preprocess Alignment Done!"
cd "$current_dir"

#Convert fasta2phylip
if ! bash convert_mafft_alignment2phylip.sh ${virus}; then
    echo "Convert Fasta2phylip failed."
    cd "$current_dir"
    exit 1
fi
echo "Convert Fasta2phylip Done!"

#Build trees
if ! bash build_tree.sh ${virus} ${path_to_trees} ${path_to_alignments}; then
    echo "Tree buliding failed."
    cd "$current_dir"
    exit 1
fi
echo "Build Tree Done!"

#Modify trees 
if ! bash modify_tree.sh ${virus} ${path_to_alignments} ${path_to_trees}; then
    echo "Tree modification failed."
    cd "$current_dir"
    exit 1
fi
echo "Modify Tree Done!"

#echo "${virus} ALL DONE, READY for CODEML!"

#CODML
# Ensure the output directory exists
#mkdir -p $output_dir

#for file in ${path_to_alignments}/*${virus_name}*nuc_mafft_aln_mafft.phy; do
    #name=$(basename $file)
    #name=${name%_nuc_mafft*}
    #echo $name
    
    # Find the corresponding tree, excluding .ipynb_checkpoints directory
    #corresponding_tree=$(find $path_to_trees -type f -name "*${name}*.tree" ! -path "*/.ipynb_checkpoints/*" | grep "$name" | head -n 1)
    #corresponding_seq=$file
    #echo $file
    #echo $corresponding_tree
    
    # Check if a corresponding tree was found
    #if [[ -z "$corresponding_tree" ]]; then
        #echo "No corresponding tree found for $name"
        #continue
    #fi
    
    #ndata=$(head -n 1 "${corresponding_tree}" | awk '{print $1}')

    # Define the path for the new control file and the output file
    #ctl_path="${output_dir}/${name}_codeml.ctl"
    #outfile="${output_dir}/${name}.txt"

    # Copy the template control file to a new location and modify it
    #cp "$template_ctl_path" "$ctl_path"
    
    # Update the control file with the specific paths and values
    #sed -i "s|seqfile = .*|seqfile = ${corresponding_seq}|" "$ctl_path"
    #sed -i "s|treefile = .*|treefile = ${corresponding_tree}|" "$ctl_path"
    #sed -i "s|outfile = .*|outfile = ${outfile}|" "$ctl_path"
    #sed -i "s|\bndata\b = .*|ndata = ${ndata}|" "$ctl_path"

    # Run codeml
    #codeml "$ctl_path"
#done

# Summarize the results
#bash summarize_result.sh ${virus_name}