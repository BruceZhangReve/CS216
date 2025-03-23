#python rename_files.py /path/to/your/directory

import os
import argparse

def rename_files(directory):
    files = os.listdir(directory)

    for filename in files:
        if filename.endswith(".fasta"):
            gene_name = filename.split(".")[0]
            new_filename = f"Coronaviridae_{gene_name}_nuc_mafft_aln_raw.fasta"
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            os.rename(old_file, new_file)
            print(f"Renamed {filename} to {new_filename}")

    print("All files have been renamed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename FASTA files in a directory.")
    parser.add_argument("directory", type=str, help="The directory containing the FASTA files to rename.")
    args = parser.parse_args()

    rename_files(args.directory)
