#python check_duplicates.py /path/to/your/fasta_file.fasta

import argparse
from Bio import SeqIO
from collections import defaultdict

def find_duplicates(fasta_file):
    sequences = defaultdict(list)
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[str(record.seq)].append(record.id)
    
    duplicates = {seq: ids for seq, ids in sequences.items() if len(ids) > 1}
    
    if duplicates:
        print("Found duplicate sequences:")
        for seq, ids in duplicates.items():
            print(f"Sequence: {seq}")
            print(f"IDs: {', '.join(ids)}\n")
    else:
        print("No duplicate sequences found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find duplicate sequences in a FASTA file.")
    parser.add_argument("fasta_file", type=str, help="The FASTA file to check for duplicates.")
    args = parser.parse_args()
    
    find_duplicates(args.fasta_file)

