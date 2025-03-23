import torch
import os
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split

vocabulary = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "AAA", "AAT", "AAC", "AAG", "ATA", "ATT", "ATC", "ATG",
    "ACA", "ACT", "ACC", "ACG", "AGA", "AGT", "AGC", "AGG",
    "TAA", "TAT", "TAC", "TAG", "TTA", "TTT", "TTC", "TTG",
    "TCA", "TCT", "TCC", "TCG", "TGA", "TGT", "TGC", "TGG",
    "CAA", "CAT", "CAC", "CAG", "CTA", "CTT", "CTC", "CTG",
    "CCA", "CCT", "CCC", "CCG", "CGA", "CGT", "CGC", "CGG",
    "GAA", "GAT", "GAC", "GAG", "GTA", "GTT", "GTC", "GTG",
    "GCA", "GCT", "GCC", "GCG", "GGA", "GGT", "GGC", "GGG"
]

vocab_to_index = {token: idx for idx, token in enumerate(vocabulary)}

def tokenize_sequence(sequence, max_length=512, k=3):
    tokens = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k].upper()
        tokens.append(vocab_to_index.get(kmer, 1))  
    
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens += [0] * (max_length - len(tokens)) 
    
    return tokens

def process_alignment_file(alignment_file, max_length=512):
    """处理比对文件为token ID序列"""
    all_sequences = []
    accessions = []  
    
    for record in SeqIO.parse(alignment_file, "fasta"):
        sequence = str(record.seq)
        token_ids = tokenize_sequence(sequence, max_length)
        all_sequences.append(token_ids)
        accessions.append(record.id)  
        
    if not all_sequences:
        raise ValueError("No valid sequences processed")
    
    return torch.tensor(all_sequences, dtype=torch.long), accessions  

def split_and_save(data_tensor, accessions, output_dir, split_ratio=(0.7, 0.15, 0.15), random_seed=42):
    os.makedirs(output_dir, exist_ok=True)
    
    assert len(split_ratio) == 3, "Split ratio must contain 3 elements"
    assert all(r > 0 for r in split_ratio), "All ratios must be positive"
    assert abs(sum(split_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    mask = (data_tensor != 0).long()
    
    indices = np.arange(data_tensor.shape[0])
    train_idx, temp_idx = train_test_split(
        indices, 
        test_size=1 - split_ratio[0],
        random_state=random_seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=split_ratio[2]/(split_ratio[1]+split_ratio[2]),
        random_state=random_seed
    )
    
    torch.save({
        'vector': data_tensor[train_idx],
        'mask': mask[train_idx],
        'accession': [accessions[i] for i in train_idx]  
    }, os.path.join(output_dir, "train.pt"))
    
    torch.save({
        'vector': data_tensor[val_idx],
        'mask': mask[val_idx],
        'accession': [accessions[i] for i in val_idx]
    }, os.path.join(output_dir, "val.pt"))
    
    torch.save({
        'vector': data_tensor[test_idx],
        'mask': mask[test_idx],
        'accession': [accessions[i] for i in test_idx]
    }, os.path.join(output_dir, "test.pt"))
    
    print(f"Data Saved at {output_dir}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

if __name__ == "__main__":
    alignment_file = "./data/alignments/OPG027.aln"
    output_dir = "./MPXV/OPG027"
    max_length = 1024
    random_seed = 42
    split_ratio = (0.7, 0.15, 0.15)  
    
    tokenized_data, accessions = process_alignment_file(alignment_file, max_length)
    
    split_and_save(
        tokenized_data,
        accessions,  
        output_dir,
        split_ratio=split_ratio,
        random_seed=random_seed
    )