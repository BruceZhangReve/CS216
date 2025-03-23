import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import json
import os

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

class MLPVAE(nn.Module):
    def __init__(self, 
                 vocab_size=len(vocabulary),
                 seq_length=512,
                 emb_dim=32,
                 hidden_dim=256,
                 latent_dim=16):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(seq_length * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim//2, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_length * emb_dim)
        )

        self.fc_out = nn.Linear(emb_dim, vocab_size)
        
        self.seq_length = seq_length
        self.emb_dim = emb_dim

    def encode(self, x):
        # x: [batch_size, seq_length]
        embedded = self.embedding(x)  # [batch, seq, emb]
        flattened = embedded.view(x.size(0), -1)  # [batch, seq*emb]
        hidden = self.encoder(flattened)
        return self.fc_mu(hidden), self.fc_var(hidden)
    
    def decode(self, z):
        # z: [batch, latent_dim]
        reconstructed = self.decoder(z)
        reconstructed = reconstructed.view(-1, self.seq_length, self.emb_dim)
        return self.fc_out(reconstructed)  # [batch, seq, vocab_size]
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        
        z = self.reparameterize(mu, logvar)

        recon_logits = self.decode(z)
        
        return recon_logits, mu, logvar


def train_model(train_loader, device, output_dir, 
                epochs=100, lr=1e-3, 
                save_interval=10, latent_dim=16):
    model = MLPVAE(
        vocab_size=len(vocabulary),
        seq_length=1024,
        emb_dim=32,
        hidden_dim=256,
        latent_dim=latent_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "embeddings"), exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for vectors, masks in train_loader:
            vectors = vectors.to(device)
            masks = masks.to(device)
            
            recon_logits, mu, logvar = model(vectors)
            
            recon_loss = F.cross_entropy(
                recon_logits.view(-1, len(vocabulary)),
                vectors.view(-1),
                ignore_index=0,
                reduction='sum'
            )
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if (epoch+1) % save_interval == 0:
            checkpoint = {
                'epoch': epoch+1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(
                checkpoint,
                os.path.join(output_dir, "checkpoints", f"checkpoint_{epoch+1}.pt")
            )
    
    print("Saving final embeddings...")
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for vectors, _ in train_loader:
            vectors = vectors.to(device)
            mu, _ = model.encode(vectors)
            all_embeddings.append(mu.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(all_embeddings, os.path.join(output_dir, "embeddings", "latent_embeddings.pt"))
    
    with open(os.path.join(output_dir, "loss_history.json"), 'w') as f:
        json.dump(loss_history, f)
    
    return model


if __name__ == "__main__":
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    input_dir = "./MPXV/OPG027"
    output_dir = "./MPXV_latent/OPG027"
    
    training_params = {
        'epochs': 300,
        'lr': 1e-4,
        'save_interval': 50,
        'latent_dim': 32
    }
    
    train_data = torch.load(os.path.join(input_dir, "train.pt"))
    vectors = train_data['vector']
    masks = train_data['mask']
    
    dataset = TensorDataset(vectors, masks)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)  # 增大batch_size
    
    trained_model = train_model(
        train_loader, 
        device, 
        output_dir,
        **training_params
    )