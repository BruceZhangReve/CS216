#python check_alignment.py /path/to/your/directory

import os
import argparse
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def check_alignment(file_path):
    try:
        alignment = AlignIO.read(file_path, "fasta")
        lengths = [len(record.seq) for record in alignment]
        if len(set(lengths)) != 1:
            print(f"Alignment error in {file_path}: sequences have different lengths {lengths}")
        else:
            print(f"{file_path} is correctly aligned")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def main(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".fasta"):
            file_path = os.path.join(directory, filename)
            check_alignment(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the alignment of FASTA files in a directory.")
    parser.add_argument("directory", type=str, help="The directory containing FASTA files.")
    args = parser.parse_args()
    
    main(args.directory)

