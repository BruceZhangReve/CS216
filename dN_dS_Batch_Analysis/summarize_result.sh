#!/bin/bash
virus=$1
result_dir="./result"
output_file="${result_dir}/Summary_${virus}.txt" #output file name

#bash summarize_result.sh "Human_mastadenovirus_C"

#Make sure output file is empty
> "$output_file"


echo "Starting to process files..."

for file in ${result_dir}/${virus}*.txt; do
    echo "Currently processing: $file"

    #Skip summary.txt
    if [[ "$file" == "$output_file" ]]; then
        echo "Skipping summary file itself: $file"
        continue
    fi

    #Check whether the txt is empty
    if [ ! -s "$file" ]; then
        echo "Skipping empty file: $file"
        continue
    fi

    #scan from back to head(faster)
    found_line=$(tac "$file" | grep -m 1 "omega (dN/dS) =")
    if [ -n "$found_line" ]; then
        echo "Found line: $found_line"
        echo "$file: $found_line" >> "$output_file"
    fi
done

echo "Summary has been saved to $output_file."







