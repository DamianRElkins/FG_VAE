import ast
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from scipy import sparse
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as L
import wandb

def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the VAE loss, including reconstruction (BCE) and KL divergence, with optional beta weighting.
    Returns total loss, BCE, and KLD.
    """
    bce = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)  # Average over batch
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    return (bce + beta * kld) , bce , kld

def save_model(model, output_dir, model_name):
    """Saves a given model to output directory"""
    # Save the trained model
    if not os.path.exists(f'{output_dir}'):
        os.makedirs(f'{output_dir}')
    torch.save(model.state_dict(), f'{output_dir}/{model_name}_vae_model.pth')
    print(f"{model_name} model trained and saved as '{output_dir}/{model_name}_vae_model.pth'")

class FingerprintDataset(Dataset):
    def __init__(self, df, sparse=True):
        # Store the DataFrame directly. It now only contains sparse data columns.
        self.df = df
        self.sparse = sparse

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.sparse:
            # Parse string representations back to lists for the current row
            fg_data_values = ast.literal_eval(row['fg_data_values'])
            fg_indices = ast.literal_eval(row['fg_indices'])
            fg_indptr = ast.literal_eval(row['fg_indptr'])

            fp_data_values = ast.literal_eval(row['fp_data_values'])
            fp_indices = ast.literal_eval(row['fp_indices'])
            fp_indptr = ast.literal_eval(row['fp_indptr'])

            # Reconstruct sparse matrix and convert to dense array for the current row
            fg_array = sparse.csr_matrix(
                (fg_data_values, fg_indices, fg_indptr),
                shape=(1, row['fg_length'])
            ).toarray().flatten()

            fingerprint_array = sparse.csr_matrix(
                (fp_data_values, fp_indices, fp_indptr),
                shape=(1, row['fp_length'])
            ).toarray().flatten()

            # Convert to tensors
            fingerprint = torch.tensor(fingerprint_array, dtype=torch.float32)
            fg_vector = torch.tensor(fg_array, dtype=torch.float32)

        else:
            # Make sure they are numpy arrays
            fingerprint_array = np.array(row['fingerprint_array'])
            fg_array = np.array(row['fg_array'])
            # Handle dense representations
            fingerprint = torch.tensor(fingerprint_array, dtype=torch.float32)
            fg_vector = torch.tensor(fg_array, dtype=torch.float32)

        return fingerprint, fg_vector, idx

# -------- Base MODEL --------
class BaseVAE(L.LightningModule):
    def __init__(self, input_dim, latent_dim, encoder_hidden_dims, decoder_hidden_dims):
        super(BaseVAE, self).__init__()
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim * 2))  # Output mu and log_var
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid()) 
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    

    

# -------- Base Model Trainer --------    
class BaseVAETrainer(L.LightningModule):
    def __init__(self, input_dim, encoder_hidden_dim, decoder_hidden_dim, latent_dim, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.model = BaseVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden_dim,
            decoder_hidden_dims=decoder_hidden_dim
        )
        self.learning_rate = learning_rate
        self.weights = torch.ones(input_dim)  # Initialize weights for BCE loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        recon_x, mu, log_var = self(x)
        beta = min(1.5, self.current_epoch / 2)  # Gradually increase beta
        loss, bce, kld = vae_loss(recon_x, x, mu, log_var, beta=beta)
        self.log('train_loss', loss)
        self.log('train_bce', bce)
        self.log('train_kld', kld)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        recon_x, mu, log_var = self(x)
        loss, bce, kld = vae_loss(recon_x, x, mu, log_var)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_bce', bce, on_epoch=True, prog_bar=True)
        self.log('val_kld', kld, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, _, _ = batch
        recon_x, mu, log_var = self(x)
        loss, bce, kld = vae_loss(recon_x, x, mu, log_var)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_bce', bce, on_epoch=True, prog_bar=True)
        self.log('test_kld', kld, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

# -------- Base Model Training Function --------
def train_base_model(dataset, input_dim, latent_dim, fg_dim, encoder_hidden_dims, decoder_hidden_dims, batch_size=64, learning_rate=1e-3, max_epochs=50, sparse=True):
    # Split dataset into train, validation, and test sets
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

    # Create datasets
    train_dataset = FingerprintDataset(train_data, sparse=sparse)
    val_dataset = FingerprintDataset(val_data, sparse=sparse)
    test_dataset = FingerprintDataset(test_data, sparse=sparse)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True)

    # Initialize model
    model = BaseVAETrainer(
        input_dim=input_dim,
        encoder_hidden_dim=encoder_hidden_dims,
        decoder_hidden_dim=decoder_hidden_dims,
        latent_dim=latent_dim,
        learning_rate=learning_rate
    )

    # Initialize Wandb logger
    wandb_logger = WandbLogger(project=f'fg_vae_{fg_dim}', log_model=True)

    # Save model checkpoints
    checkpoint_callback = L.pytorch_lightning.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/fg_bvae_{fg_dim}_{latent_dim}',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    # Train the model
    trainer = L.Trainer(
        max_epochs=max_epochs,
        val_check_interval=0.5,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    trainer.test(model, test_loader)

    wandb.finish()

    return model

# -------- CVAE Model --------
class ConditionalVAE(L.LightningModule):
    def __init__(self, fingerprint_dim, fg_dim, latent_dim, encoder_hidden_dims, decoder_hidden_dims):
        super(ConditionalVAE, self).__init__()
        
        self.fingerprint_dim = fingerprint_dim
        self.fg_dim = fg_dim
        self.latent_dim = latent_dim
        
        # Encoder: fingerprint + FG vector
        encoder_layers = []
        prev_dim = fingerprint_dim + fg_dim
        for h_dim in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim * 2))  # outputs mu and log_var
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: latent + FG vector
        decoder_layers = []
        prev_dim = latent_dim + fg_dim
        for h_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, fingerprint_dim))
        decoder_layers.append(nn.Sigmoid())  # Predict fingerprint (multi-label output)
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x, fg):
        h = self.encoder(torch.cat([x, fg], dim=-1))
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, fg):
        return self.decoder(torch.cat([z, fg], dim=-1))

    def forward(self, x, fg):
        mu, log_var = self.encode(x, fg)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, fg), mu, log_var

# -------- CVAE Trainer --------
class ConditionalVAETrainer(L.LightningModule):
    def __init__(self, fingerprint_dim, fg_dim, encoder_hidden_dims, decoder_hidden_dims, latent_dim, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.model = ConditionalVAE(
            fingerprint_dim=fingerprint_dim,
            fg_dim=fg_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims
        )
        self.learning_rate = learning_rate

    def forward(self, x, fg):
        return self.model(x, fg)


    def training_step(self, batch, batch_idx):
        x, fg, _ = batch
        recon_x, mu, log_var = self(x, fg)
        beta = min(1.5, self.current_epoch / 2)  # KL warmup
        loss, bce, kld = vae_loss(recon_x, x, mu, log_var, beta=beta)
        self.log('train_loss', loss)
        self.log('train_bce', bce)
        self.log('train_kld', kld)
        return loss

    def validation_step(self, batch, batch_idx):
        x, fg, _ = batch
        recon_x, mu, log_var = self(x, fg)
        loss, bce, kld = vae_loss(recon_x, x, mu, log_var)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_bce', bce, on_epoch=True, prog_bar=True)
        self.log('val_kld', kld, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, fg, _ = batch
        recon_x, mu, log_var = self(x, fg)
        loss, bce, kld = vae_loss(recon_x, x, mu, log_var)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_bce', bce, on_epoch=True, prog_bar=True)
        self.log('test_kld', kld, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

# -------- CVAE Training Function --------
def train_conditional_vae(dataset, fingerprint_dim, fg_dim, latent_dim, encoder_hidden_dims, decoder_hidden_dims, batch_size=64, learning_rate=1e-3, max_epochs=50, sparse=True):
    # Split dataset
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.2, random_state=42)

    # Datasets & loaders
    train_dataset = FingerprintDataset(train_data, sparse=sparse)
    val_dataset = FingerprintDataset(val_data, sparse=sparse)
    test_dataset = FingerprintDataset(test_data, sparse=sparse)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True)

    # Initialize model
    model = ConditionalVAETrainer(
        fingerprint_dim=fingerprint_dim,
        fg_dim=fg_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        latent_dim=latent_dim,
        learning_rate=learning_rate
    )

    # WandB logger
    wandb_logger = WandbLogger(project=f'fg_cvae_{fg_dim}', log_model=True)

    # Trainer
    # Save checkpoints in .ckpt format
    checkpoint_callback = L.pytorch_lightning.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/fg_cvae_{fg_dim}_{latent_dim}',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        val_check_interval=0.5,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    wandb.finish()

    return model

# -------- CSVAE Model --------
class ConditionalSubspaceVAE(L.LightningModule):
    def __init__(self, fingerprint_dim, fg_dim, latent_dim_z, latent_dim_w, encoder_hidden_dims_z, encoder_hidden_dims_w, decoder_hidden_dims, adversarial_hidden_dims):
        super(ConditionalSubspaceVAE, self).__init__()
        self.fingerprint_dim = fingerprint_dim
        self.fg_dim = fg_dim
        self.latent_dim_z = latent_dim_z
        self.latent_dim_w = latent_dim_w

        # Encoder for latent space Z (only depends on x)
        encoder_z_layers = []
        prev_dim = fingerprint_dim
        for h_dim in encoder_hidden_dims_z:
            encoder_z_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_z_layers.append(nn.ReLU())
            prev_dim = h_dim
        encoder_z_layers.append(nn.Linear(prev_dim, latent_dim_z * 2))
        self.encoder_z = nn.Sequential(*encoder_z_layers)

        # Encoder for latent space W (depends on x and y)
        encoder_w_layers = []
        prev_dim = fingerprint_dim + fg_dim
        for h_dim in encoder_hidden_dims_w:
            encoder_w_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_w_layers.append(nn.ReLU())
            prev_dim = h_dim
        encoder_w_layers.append(nn.Linear(prev_dim, latent_dim_w * 2))
        self.encoder_w = nn.Sequential(*encoder_w_layers)

        # Decoder (depends on z and w)
        decoder_layers = []
        prev_dim = latent_dim_z + latent_dim_w
        for h_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, fingerprint_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

        # Adversarial Network (classifier) to predict y from z
        adversarial_layers = []
        prev_dim = latent_dim_z
        for h_dim in adversarial_hidden_dims:
            adversarial_layers.append(nn.Linear(prev_dim, h_dim))
            adversarial_layers.append(nn.ReLU())
            prev_dim = h_dim
        adversarial_layers.append(nn.Linear(prev_dim, fg_dim))
        adversarial_layers.append(nn.Sigmoid())
        self.adversarial_network = nn.Sequential(*adversarial_layers)

    def encode(self, x, fg):
        mu_z, log_var_z = self.encoder_z(x).chunk(2, dim=-1)
        mu_w, log_var_w = self.encoder_w(torch.cat([x, fg], dim=-1)).chunk(2, dim=-1)
        return mu_z, log_var_z, mu_w, log_var_w

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, w):
        return self.decoder(torch.cat([z, w], dim=-1))

    def classify_z(self, z):
        return self.adversarial_network(z)

    def forward(self, x, fg):
        mu_z, log_var_z, mu_w, log_var_w = self.encode(x, fg)
        z = self.reparameterize(mu_z, log_var_z)
        w = self.reparameterize(mu_w, log_var_w)
        recon_x = self.decode(z, w)
        fg_pred = self.classify_z(z)
        return recon_x, fg_pred, mu_z, log_var_z, mu_w, log_var_w

# -------- CSVAE Trainer --------
class ConditionalSubspaceVAETrainer(L.LightningModule):
    def __init__(self, fingerprint_dim, fg_dim, latent_dim_z, latent_dim_w, encoder_hidden_dims_z, encoder_hidden_dims_w, decoder_hidden_dims, adversarial_hidden_dims, learning_rate, beta1=1.0, beta2=1.0, beta3=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # Enable manual optimization
        self.model = ConditionalSubspaceVAE(
            fingerprint_dim=fingerprint_dim,
            fg_dim=fg_dim,
            latent_dim_z=latent_dim_z,
            latent_dim_w=latent_dim_w,
            encoder_hidden_dims_z=encoder_hidden_dims_z,
            encoder_hidden_dims_w=encoder_hidden_dims_w,
            decoder_hidden_dims=decoder_hidden_dims,
            adversarial_hidden_dims=adversarial_hidden_dims
        )
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.reconstruction_loss_func = nn.BCELoss(reduction='sum')
        self.adversarial_loss_func = nn.BCELoss(reduction='sum')

    def forward(self, x, fg):
        return self.model(x, fg)

    def training_step(self, batch, batch_idx):
        opt_vae, opt_adv = self.optimizers()
        x, fg, _ = batch

        recon_x, fg_pred, mu_z, log_var_z, mu_w, log_var_w = self(x, fg)
        z = self.model.reparameterize(mu_z, log_var_z)

        # ----- losses (use sums) -----
        recon_loss = self.reconstruction_loss_func(recon_x, x) / x.size(0)  # sum averaged over batches
        kld_z = -0.5 * (1 + log_var_z - mu_z.pow(2) - log_var_z.exp()).sum() / x.size(0)
        kld_w = -0.5 * (1 + log_var_w - mu_w.pow(2) - log_var_w.exp()).sum() / x.size(0)
        vae_loss_val = recon_loss + kld_z + kld_w

        # M2: maximize entropy => minimize (-H) for Bernoulli dims
        eps = 1e-10
        bernoulli_negH = (fg_pred * torch.log(fg_pred + eps) + (1 - fg_pred) * torch.log(1 - fg_pred + eps)).sum() / x.size(0)
        m2_loss = bernoulli_negH  # this is -H

        total_vae_loss = self.beta1 * vae_loss_val + self.beta2 * m2_loss

        # ----- VAE step -----
        opt_vae.zero_grad()
        self.manual_backward(total_vae_loss) 
        opt_vae.step()

        # ----- adversary step (detach z) -----
        z_detached = z.detach()
        fg_pred_adv = self.model.classify_z(z_detached)
        n_loss = self.adversarial_loss_func(fg_pred_adv, fg) / x.size(0)
        total_adv_loss = self.beta3 * n_loss

        opt_adv.zero_grad()
        self.manual_backward(total_adv_loss)
        opt_adv.step()

        # logs
        self.log('train_total_vae', total_vae_loss, on_epoch=True)
        self.log('train_recon', recon_loss, on_epoch=True)
        self.log('train_kld_z', kld_z, on_epoch=True)
        self.log('train_kld_w', kld_w, on_epoch=True)
        self.log('train_m2_negH', m2_loss, on_epoch=True)
        self.log('train_adv', total_adv_loss, on_epoch=True)


    def validation_step(self, batch, batch_idx):
        x, fg, _ = batch
        recon_x, fg_pred, mu_z, log_var_z, mu_w, log_var_w = self(x, fg)
        recon_loss = self.reconstruction_loss_func(recon_x, x) / x.size(0)
        kld_z = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp()) / x.size(0)
        kld_w = -0.5 * torch.sum(1 + log_var_w - mu_w.pow(2) - log_var_w.exp()) / x.size(0)
        vae_loss_val = recon_loss + kld_z + kld_w
        n_loss = self.adversarial_loss_func(fg_pred, fg) / x.size(0)
        self.log('val_vae_loss', vae_loss_val, on_epoch=True, prog_bar=True)
        self.log('val_adv_loss', n_loss, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', recon_loss, on_epoch=True)
        self.log('val_kld_z', kld_z, on_epoch=True)
        self.log('val_kld_w', kld_w, on_epoch=True)
        return vae_loss_val + n_loss
    
    def test_step(self, batch, batch_idx):
        x, fg, _ = batch
        recon_x, fg_pred, mu_z, log_var_z, mu_w, log_var_w = self(x, fg)
        recon_loss = self.reconstruction_loss_func(recon_x, x) / x.size(0)
        kld_z = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp()) / x.size(0)
        kld_w = -0.5 * torch.sum(1 + log_var_w - mu_w.pow(2) - log_var_w.exp()) / x.size(0)
        vae_loss_val = recon_loss + kld_z + kld_w
        n_loss = self.adversarial_loss_func(fg_pred, fg) / x.size(0)
        self.log('test_vae_loss', vae_loss_val, on_epoch=True, prog_bar=True)
        self.log('test_adv_loss', n_loss, on_epoch=True, prog_bar=True)
        self.log('test_recon_loss', recon_loss, on_epoch=True)
        self.log('test_kld_z', kld_z, on_epoch=True)
        self.log('test_kld_w', kld_w, on_epoch=True)
        return vae_loss_val + n_loss

    def configure_optimizers(self):
        optimizer_vae = optim.Adam(
            list(self.model.encoder_z.parameters()) +
            list(self.model.encoder_w.parameters()) +
            list(self.model.decoder.parameters()),
            lr=self.learning_rate
        )
        optimizer_adv = optim.Adam(
            self.model.adversarial_network.parameters(),
            lr=self.learning_rate
        )
        return [optimizer_vae, optimizer_adv]


# -------- CSVAE Training Function --------
def train_conditional_subspace_vae(dataset, fingerprint_dim, fg_dim, latent_dim_z, latent_dim_w, encoder_hidden_dims_z, encoder_hidden_dims_w, decoder_hidden_dims, adversarial_hidden_dims, batch_size=64, learning_rate=1e-3, max_epochs=50, sparse=True):
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.2, random_state=42)
    train_dataset = FingerprintDataset(train_data, sparse=sparse)
    val_dataset = FingerprintDataset(val_data, sparse=sparse)
    test_dataset = FingerprintDataset(test_data, sparse=sparse)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True)
    model = ConditionalSubspaceVAETrainer(
        fingerprint_dim=fingerprint_dim,
        fg_dim=fg_dim,
        latent_dim_z=latent_dim_z,
        latent_dim_w=latent_dim_w,
        encoder_hidden_dims_z=encoder_hidden_dims_z,
        encoder_hidden_dims_w=encoder_hidden_dims_w,
        decoder_hidden_dims=decoder_hidden_dims,
        adversarial_hidden_dims=adversarial_hidden_dims,
        learning_rate=learning_rate
    )
    wandb_logger = WandbLogger(project=f'fg_csvae_{fg_dim}', log_model=True)
    checkpoint_callback = L.pytorch_lightning.callbacks.ModelCheckpoint(
        monitor='val_vae_loss',
        dirpath=f'checkpoints/fg_csvae_{fg_dim}_{latent_dim_z}',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        val_check_interval=0.5,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    wandb.finish()
    return model

# -------- DISCoVeR Model --------
class DiscoverVAE(L.LightningModule):
    def __init__(self, fingerprint_dim, fg_dim, latent_dim_z, latent_dim_w, encoder_hidden_dims_z, encoder_hidden_dims_w, decoder_hidden_dims, decoder_z_hidden_dims, adversarial_hidden_dims):
        super(DiscoverVAE, self).__init__()
        self.fingerprint_dim = fingerprint_dim
        self.fg_dim = fg_dim
        self.latent_dim_z = latent_dim_z
        self.latent_dim_w = latent_dim_w

        # Encoder for latent space Z (only depends on x)
        encoder_z_layers = []
        prev_dim = fingerprint_dim
        for h_dim in encoder_hidden_dims_z:
            encoder_z_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_z_layers.append(nn.ReLU())            
            prev_dim = h_dim
        encoder_z_layers.append(nn.Linear(prev_dim, latent_dim_z * 2))
        self.encoder_z = nn.Sequential(*encoder_z_layers)

        # Encoder for latent space W (depends on x and y)
        encoder_w_layers = []
        prev_dim = fingerprint_dim + fg_dim
        for h_dim in encoder_hidden_dims_w:
            encoder_w_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_w_layers.append(nn.ReLU())
            prev_dim = h_dim
        encoder_w_layers.append(nn.Linear(prev_dim, latent_dim_w * 2))
        self.encoder_w = nn.Sequential(*encoder_w_layers)

        # Decoder 1 (full reconstruction, depends on z and w)
        decoder_layers = []
        prev_dim = latent_dim_z + latent_dim_w
        for h_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, fingerprint_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

        # Decoder 2 (lightweight reconstruction, depends on z only)
        decoder_z_layers = []
        prev_dim = latent_dim_z
        for h_dim in decoder_z_hidden_dims:
            decoder_z_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_z_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_z_layers.append(nn.Linear(prev_dim, fingerprint_dim))
        decoder_z_layers.append(nn.Sigmoid())
        self.decoder_z = nn.Sequential(*decoder_z_layers)

        # Adversarial Network (classifier) to predict y from z
        adversarial_layers = []
        prev_dim = fingerprint_dim
        for h_dim in adversarial_hidden_dims:
            adversarial_layers.append(nn.Linear(prev_dim, h_dim))
            adversarial_layers.append(nn.ReLU())
            adversarial_layers.append(nn.Dropout(0.1))
            prev_dim = h_dim
        adversarial_layers.append(nn.Linear(prev_dim, fg_dim))
        adversarial_layers.append(nn.Sigmoid())
        self.adversarial_network = nn.Sequential(*adversarial_layers)


    def encode(self, x, fg):
        mu_z, log_var_z = self.encoder_z(x).chunk(2, dim=-1)
        mu_w, log_var_w = self.encoder_w(torch.cat([x, fg], dim=-1)).chunk(2, dim=-1)
        return mu_z, log_var_z, mu_w, log_var_w

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, w):
        return self.decoder(torch.cat([z, w], dim=-1))

    def decode_z(self, z):
        return self.decoder_z(z)

    def classify_x_hat(self, x_hat):
        return self.adversarial_network(x_hat)

    def forward(self, x, fg):
        mu_z, log_var_z, mu_w, log_var_w = self.encode(x, fg)
        z = self.reparameterize(mu_z, log_var_z)
        w = self.reparameterize(mu_w, log_var_w)
        recon_x_full = self.decode(z, w)
        recon_x_hat = self.decode_z(z)
        fg_pred = self.classify_x_hat(recon_x_hat)
        return recon_x_full, recon_x_hat, fg_pred, mu_z, log_var_z, mu_w, log_var_w


# -------- DISCoVeR Trainer --------
class DiscoverVAETrainer(L.LightningModule):
    def __init__(self, fingerprint_dim, fg_dim, latent_dim_z, latent_dim_w, encoder_hidden_dims_z, encoder_hidden_dims_w, decoder_hidden_dims, decoder_z_hidden_dims, adversarial_hidden_dims, learning_rate, beta1=0.7, beta2=0.7, beta3=0.2, beta4=0.8, beta5=0.3):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.model = DiscoverVAE(
            fingerprint_dim=fingerprint_dim,
            fg_dim=fg_dim,
            latent_dim_z=latent_dim_z,
            latent_dim_w=latent_dim_w,
            encoder_hidden_dims_z=encoder_hidden_dims_z,
            encoder_hidden_dims_w=encoder_hidden_dims_w,
            decoder_hidden_dims=decoder_hidden_dims,
            decoder_z_hidden_dims=decoder_z_hidden_dims,
            adversarial_hidden_dims=adversarial_hidden_dims
        )
        self.learning_rate = learning_rate
        self.rec_w = beta1
        self.kld_z_w = beta2
        self.kld_w_w = beta3
        self.adv_w = beta4
        self.rec_z_w = beta5

        self.reconstruction_loss_func = nn.BCELoss(reduction='sum')
        self.adversarial_loss_func = nn.BCELoss(reduction='sum')

    def forward(self, x, fg):
        return self.model(x, fg)

    def training_step(self, batch, batch_idx):
        opt_vae, opt_adv = self.optimizers()
        x, fg, _ = batch

        recon_x_full, recon_x_hat, fg_pred, mu_z, log_var_z, mu_w, log_var_w = self(x, fg)

        # ----- VAE losses (main step) -----
        recon_loss_full = self.reconstruction_loss_func(recon_x_full, x) / x.size(0)
        recon_loss_hat = self.reconstruction_loss_func(recon_x_hat, x) / x.size(0)
        kld_z = -0.5 * (1 + log_var_z - mu_z.pow(2) - log_var_z.exp()).sum() / x.size(0)
        kld_w = -0.5 * (1 + log_var_w - mu_w.pow(2) - log_var_w.exp()).sum() / x.size(0)

        # M2: maximize entropy => minimize (-H) for Bernoulli dims
        neg_adv_loss = -self.adversarial_loss_func(fg_pred, fg) / x.size(0)

        total_vae_loss = (self.rec_w * recon_loss_full + 
                          self.kld_z_w * kld_z + 
                          self.kld_w_w * kld_w +
                          self.adv_w * neg_adv_loss + 
                          self.rec_z_w * recon_loss_hat)


        # ----- VAE step -----
        opt_vae.zero_grad()
        self.manual_backward(total_vae_loss) 
        opt_vae.step()

        # ----- Adversary step (detach z) -----
        rec_hat_detached = recon_x_hat.detach()
        fg_pred_adv = self.model.classify_x_hat(rec_hat_detached)
        adv_loss_val = self.adversarial_loss_func(fg_pred_adv, fg) / x.size(0)

        opt_adv.zero_grad()
        self.manual_backward(adv_loss_val)
        opt_adv.step()

        # logs
        self.log('train_total_vae', total_vae_loss, on_epoch=True)
        self.log('train_recon_full', recon_loss_full, on_epoch=True)
        self.log('train_recon_z', recon_loss_hat, on_epoch=True)
        self.log('train_kld_z', kld_z, on_epoch=True)
        self.log('train_kld_w', kld_w, on_epoch=True)
        self.log('train_adv_vae', neg_adv_loss, on_epoch=True)
        self.log('train_adv_disc', adv_loss_val, on_epoch=True)


    def validation_step(self, batch, batch_idx):
        x, fg, _ = batch
        recon_x_full, recon_x_hat, fg_pred, mu_z, log_var_z, mu_w, log_var_w = self(x, fg)
        recon_loss_full = self.reconstruction_loss_func(recon_x_full, x) / x.size(0)
        recon_loss_hat = self.reconstruction_loss_func(recon_x_hat, x) / x.size(0)
        kld_z = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp()) / x.size(0)
        kld_w = -0.5 * torch.sum(1 + log_var_w - mu_w.pow(2) - log_var_w.exp()) / x.size(0)
        adv_loss_val = self.adversarial_loss_func(fg_pred, fg) / x.size(0)
        total_loss = self.rec_w * recon_loss_full + self.kld_z_w * kld_z + self.kld_w_w * kld_w + self.rec_z_w * recon_loss_hat - self.adv_w * adv_loss_val

        self.log('val_total_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_recon_full', recon_loss_full, on_epoch=True)
        self.log('val_recon_z', recon_loss_hat, on_epoch=True)
        self.log('val_kld_z', kld_z, on_epoch=True)
        self.log('val_kld_w', kld_w, on_epoch=True)
        self.log('val_adv_disc', adv_loss_val, on_epoch=True)
        return total_loss
    
    def test_step(self, batch, batch_idx):
        x, fg, _ = batch
        recon_x_full, recon_x_hat, fg_pred, mu_z, log_var_z, mu_w, log_var_w = self(x, fg)
        recon_loss_full = self.reconstruction_loss_func(recon_x_full, x) / x.size(0)
        recon_loss_hat = self.reconstruction_loss_func(recon_x_hat, x) / x.size(0)
        kld_z = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp()) / x.size(0)
        kld_w = -0.5 * torch.sum(1 + log_var_w - mu_w.pow(2) - log_var_w.exp()) / x.size(0)
        adv_loss_val = self.adversarial_loss_func(fg_pred, fg) / x.size(0)
        total_loss = self.rec_w * recon_loss_full + self.kld_z_w * kld_z + self.kld_w_w * kld_w + self.rec_z_w * recon_loss_hat - self.adv_w * adv_loss_val

        self.log('test_total_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('test_recon_full', recon_loss_full, on_epoch=True)
        self.log('test_recon_z', recon_loss_hat, on_epoch=True)
        self.log('test_kld_z', kld_z, on_epoch=True)
        self.log('test_kld_w', kld_w, on_epoch=True)
        self.log('test_adv_disc', adv_loss_val, on_epoch=True)
        return total_loss

    def configure_optimizers(self):
        optimizer_vae = optim.Adam(
            list(self.model.encoder_z.parameters()) +
            list(self.model.encoder_w.parameters()) +
            list(self.model.decoder.parameters()) +
            list(self.model.decoder_z.parameters()),
            lr=self.learning_rate
        )
        optimizer_adv = optim.Adam(
            self.model.adversarial_network.parameters(),
            lr=self.learning_rate
        )
        return [optimizer_vae, optimizer_adv]


# -------- DISCoVeR Training Function --------
def train_discover_vae(dataset, fingerprint_dim, fg_dim, latent_dim_z, latent_dim_w, encoder_hidden_dims_z, encoder_hidden_dims_w, decoder_hidden_dims, decoder_z_hidden_dims, adversarial_hidden_dims, batch_size=64, learning_rate=1e-3, max_epochs=50, sparse=True):
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.2, random_state=42)
    train_dataset = FingerprintDataset(train_data, sparse=sparse)
    val_dataset = FingerprintDataset(val_data, sparse=sparse)
    test_dataset = FingerprintDataset(test_data, sparse=sparse)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True)
    model = DiscoverVAETrainer(
        fingerprint_dim=fingerprint_dim,
        fg_dim=fg_dim,
        latent_dim_z=latent_dim_z,
        latent_dim_w=latent_dim_w,
        encoder_hidden_dims_z=encoder_hidden_dims_z,
        encoder_hidden_dims_w=encoder_hidden_dims_w,
        decoder_hidden_dims=decoder_hidden_dims,
        decoder_z_hidden_dims=decoder_z_hidden_dims,
        adversarial_hidden_dims=adversarial_hidden_dims,
        learning_rate=learning_rate
    )
    wandb_logger = WandbLogger(project=f'fg_discover_vae_{fg_dim}', log_model=True)

    checkpoint_callback = L.pytorch_lightning.callbacks.ModelCheckpoint(
        monitor='val_total_loss',
        dirpath=f'checkpoints/fg_dvae_{fg_dim}_{latent_dim_z}',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        val_check_interval=0.5,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    wandb.finish()
    return model


# -------- DISCoVeR Model --------
class Discover_z_VAE(L.LightningModule):
    def __init__(self, fingerprint_dim, fg_dim, latent_dim_z, latent_dim_w, encoder_hidden_dims_z, encoder_hidden_dims_w, decoder_hidden_dims, decoder_z_hidden_dims, adversarial_hidden_dims):
        super(Discover_z_VAE, self).__init__()
        self.fingerprint_dim = fingerprint_dim
        self.fg_dim = fg_dim
        self.latent_dim_z = latent_dim_z
        self.latent_dim_w = latent_dim_w

        # Encoder for latent space Z (only depends on x)
        encoder_z_layers = []
        prev_dim = fingerprint_dim
        for h_dim in encoder_hidden_dims_z:
            encoder_z_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_z_layers.append(nn.ReLU())            
            prev_dim = h_dim
        encoder_z_layers.append(nn.Linear(prev_dim, latent_dim_z * 2))
        self.encoder_z = nn.Sequential(*encoder_z_layers)

        # Encoder for latent space W (now depends only on x)
        encoder_w_layers = []
        prev_dim = fingerprint_dim  # removed fg_dim
        for h_dim in encoder_hidden_dims_w:
            encoder_w_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_w_layers.append(nn.ReLU())
            prev_dim = h_dim
        encoder_w_layers.append(nn.Linear(prev_dim, latent_dim_w * 2))
        self.encoder_w = nn.Sequential(*encoder_w_layers)


        # Decoder 1 (full reconstruction, depends on z and w)
        decoder_layers = []
        prev_dim = latent_dim_z + latent_dim_w
        for h_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, fingerprint_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

        # Decoder 2 (lightweight reconstruction, depends on z only)
        decoder_z_layers = []
        prev_dim = latent_dim_z
        for h_dim in decoder_z_hidden_dims:
            decoder_z_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_z_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_z_layers.append(nn.Linear(prev_dim, fingerprint_dim))
        decoder_z_layers.append(nn.Sigmoid())
        self.decoder_z = nn.Sequential(*decoder_z_layers)

        # Adversarial Network (classifier) to predict y from z
        adversarial_layers = []
        prev_dim = fingerprint_dim
        for h_dim in adversarial_hidden_dims:
            adversarial_layers.append(nn.Linear(prev_dim, h_dim))
            adversarial_layers.append(nn.ReLU())
            adversarial_layers.append(nn.Dropout(0.1))
            prev_dim = h_dim
        adversarial_layers.append(nn.Linear(prev_dim, fg_dim))
        adversarial_layers.append(nn.Sigmoid())
        self.adversarial_network = nn.Sequential(*adversarial_layers)


    def encode(self, x, fg=None):
        mu_z, log_var_z = self.encoder_z(x).chunk(2, dim=-1)
        # W now depends only on x
        mu_w, log_var_w = self.encoder_w(x).chunk(2, dim=-1)
        return mu_z, log_var_z, mu_w, log_var_w


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, w):
        return self.decoder(torch.cat([z, w], dim=-1))

    def decode_z(self, z):
        return self.decoder_z(z)

    def classify_x_hat(self, x_hat):
        return self.adversarial_network(x_hat)

    def forward(self, x, fg):
        mu_z, log_var_z, mu_w, log_var_w = self.encode(x, fg)
        z = self.reparameterize(mu_z, log_var_z)
        w = self.reparameterize(mu_w, log_var_w)
        recon_x_full = self.decode(z, w)
        recon_x_hat = self.decode_z(z)
        fg_pred = self.classify_x_hat(recon_x_hat)
        return recon_x_full, recon_x_hat, fg_pred, mu_z, log_var_z, mu_w, log_var_w


# -------- DISCoVeR Trainer --------
class Discover_z_VAETrainer(L.LightningModule):
    def __init__(self, fingerprint_dim, fg_dim, latent_dim_z, latent_dim_w, encoder_hidden_dims_z, encoder_hidden_dims_w, decoder_hidden_dims, decoder_z_hidden_dims, adversarial_hidden_dims, learning_rate, beta1=0.7, beta2=0.7, beta3=0.2, beta4=0.8, beta5=0.3):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.model = Discover_z_VAE(
            fingerprint_dim=fingerprint_dim,
            fg_dim=fg_dim,
            latent_dim_z=latent_dim_z,
            latent_dim_w=latent_dim_w,
            encoder_hidden_dims_z=encoder_hidden_dims_z,
            encoder_hidden_dims_w=encoder_hidden_dims_w,
            decoder_hidden_dims=decoder_hidden_dims,
            decoder_z_hidden_dims=decoder_z_hidden_dims,
            adversarial_hidden_dims=adversarial_hidden_dims
        )
        self.learning_rate = learning_rate
        self.rec_w = beta1
        self.kld_z_w = beta2
        self.kld_w_w = beta3
        self.adv_w = beta4
        self.rec_z_w = beta5

        self.reconstruction_loss_func = nn.BCELoss(reduction='sum')
        self.adversarial_loss_func = nn.BCELoss(reduction='sum')

    def forward(self, x, fg):
        return self.model(x, fg)

    def training_step(self, batch, batch_idx):
        opt_vae, opt_adv = self.optimizers()
        x, fg, _ = batch

        recon_x_full, recon_x_hat, fg_pred, mu_z, log_var_z, mu_w, log_var_w = self(x, fg)

        # ----- VAE losses (main step) -----
        recon_loss_full = self.reconstruction_loss_func(recon_x_full, x) / x.size(0)
        recon_loss_hat = self.reconstruction_loss_func(recon_x_hat, x) / x.size(0)
        kld_z = -0.5 * (1 + log_var_z - mu_z.pow(2) - log_var_z.exp()).sum() / x.size(0)
        kld_w = -0.5 * (1 + log_var_w - mu_w.pow(2) - log_var_w.exp()).sum() / x.size(0)

        # M2: maximize entropy => minimize (-H) for Bernoulli dims
        neg_adv_loss = -self.adversarial_loss_func(fg_pred, fg) / x.size(0)

        total_vae_loss = (self.rec_w * recon_loss_full + 
                          self.kld_z_w * kld_z + 
                          self.kld_w_w * kld_w +
                          self.adv_w * neg_adv_loss + 
                          self.rec_z_w * recon_loss_hat)


        # ----- VAE step -----
        opt_vae.zero_grad()
        self.manual_backward(total_vae_loss) 
        opt_vae.step()

        # ----- Adversary step (detach z) -----
        rec_hat_detached = recon_x_hat.detach()
        fg_pred_adv = self.model.classify_x_hat(rec_hat_detached)
        adv_loss_val = self.adversarial_loss_func(fg_pred_adv, fg) / x.size(0)

        opt_adv.zero_grad()
        self.manual_backward(adv_loss_val)
        opt_adv.step()

        # logs
        self.log('train_total_vae', total_vae_loss, on_epoch=True)
        self.log('train_recon_full', recon_loss_full, on_epoch=True)
        self.log('train_recon_z', recon_loss_hat, on_epoch=True)
        self.log('train_kld_z', kld_z, on_epoch=True)
        self.log('train_kld_w', kld_w, on_epoch=True)
        self.log('train_adv_vae', neg_adv_loss, on_epoch=True)
        self.log('train_adv_disc', adv_loss_val, on_epoch=True)


    def validation_step(self, batch, batch_idx):
        x, fg, _ = batch
        recon_x_full, recon_x_hat, fg_pred, mu_z, log_var_z, mu_w, log_var_w = self(x, fg)
        recon_loss_full = self.reconstruction_loss_func(recon_x_full, x) / x.size(0)
        recon_loss_hat = self.reconstruction_loss_func(recon_x_hat, x) / x.size(0)
        kld_z = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp()) / x.size(0)
        kld_w = -0.5 * torch.sum(1 + log_var_w - mu_w.pow(2) - log_var_w.exp()) / x.size(0)
        adv_loss_val = self.adversarial_loss_func(fg_pred, fg) / x.size(0)
        total_loss = self.rec_w * recon_loss_full + self.kld_z_w * kld_z + self.kld_w_w * kld_w + self.rec_z_w * recon_loss_hat - self.adv_w * adv_loss_val

        self.log('val_total_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_recon_full', recon_loss_full, on_epoch=True)
        self.log('val_recon_z', recon_loss_hat, on_epoch=True)
        self.log('val_kld_z', kld_z, on_epoch=True)
        self.log('val_kld_w', kld_w, on_epoch=True)
        self.log('val_adv_disc', adv_loss_val, on_epoch=True)
        return total_loss
    
    def test_step(self, batch, batch_idx):
        x, fg, _ = batch
        recon_x_full, recon_x_hat, fg_pred, mu_z, log_var_z, mu_w, log_var_w = self(x, fg)
        recon_loss_full = self.reconstruction_loss_func(recon_x_full, x) / x.size(0)
        recon_loss_hat = self.reconstruction_loss_func(recon_x_hat, x) / x.size(0)
        kld_z = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp()) / x.size(0)
        kld_w = -0.5 * torch.sum(1 + log_var_w - mu_w.pow(2) - log_var_w.exp()) / x.size(0)
        adv_loss_val = self.adversarial_loss_func(fg_pred, fg) / x.size(0)
        total_loss = self.rec_w * recon_loss_full + self.kld_z_w * kld_z + self.kld_w_w * kld_w + self.rec_z_w * recon_loss_hat - self.adv_w * adv_loss_val

        self.log('test_total_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('test_recon_full', recon_loss_full, on_epoch=True)
        self.log('test_recon_z', recon_loss_hat, on_epoch=True)
        self.log('test_kld_z', kld_z, on_epoch=True)
        self.log('test_kld_w', kld_w, on_epoch=True)
        self.log('test_adv_disc', adv_loss_val, on_epoch=True)
        return total_loss

    def configure_optimizers(self):
        optimizer_vae = optim.Adam(
            list(self.model.encoder_z.parameters()) +
            list(self.model.encoder_w.parameters()) +
            list(self.model.decoder.parameters()) +
            list(self.model.decoder_z.parameters()),
            lr=self.learning_rate
        )
        optimizer_adv = optim.Adam(
            self.model.adversarial_network.parameters(),
            lr=self.learning_rate
        )
        return [optimizer_vae, optimizer_adv]


# -------- DISCoVeR Training Function --------
def train_discover_z_vae(dataset, fingerprint_dim, fg_dim, latent_dim_z, latent_dim_w, encoder_hidden_dims_z, encoder_hidden_dims_w, decoder_hidden_dims, decoder_z_hidden_dims, adversarial_hidden_dims, batch_size=64, learning_rate=1e-3, max_epochs=50, sparse=True):
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.2, random_state=42)
    train_dataset = FingerprintDataset(train_data, sparse=sparse)
    val_dataset = FingerprintDataset(val_data, sparse=sparse)
    test_dataset = FingerprintDataset(test_data, sparse=sparse)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True)
    model = Discover_z_VAETrainer(
        fingerprint_dim=fingerprint_dim,
        fg_dim=fg_dim,
        latent_dim_z=latent_dim_z,
        latent_dim_w=latent_dim_w,
        encoder_hidden_dims_z=encoder_hidden_dims_z,
        encoder_hidden_dims_w=encoder_hidden_dims_w,
        decoder_hidden_dims=decoder_hidden_dims,
        decoder_z_hidden_dims=decoder_z_hidden_dims,
        adversarial_hidden_dims=adversarial_hidden_dims,
        learning_rate=learning_rate
    )
    wandb_logger = WandbLogger(project=f'fg_discover_vae_z_{fg_dim}', log_model=True)

    checkpoint_callback = L.pytorch_lightning.callbacks.ModelCheckpoint(
        monitor='val_total_loss',
        dirpath=f'checkpoints/fg_dvae_z_{fg_dim}_{latent_dim_z}',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        val_check_interval=0.5,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    wandb.finish()
    return model

# -- Latent Space Extraction --
def extract_and_save_latents(
    model_path,
    dataloader,
    model_type,
    model_class,   # LightningModule class
    device="cpu",
    map_location=None
):
    """
    Extract latent variables from a trained LightningModule (.ckpt) and save to CSV.

    Supports BaseVAE, ConditionalVAE, ConditionalSubspaceVAE, DiscoverVAE.

    Args:
        model_path (str or Path): Path to the saved .ckpt model.
        dataloader (DataLoader): DataLoader for dataset.
        model_type (str): One of ['Base', 'CVAE', 'CSVAE', 'DISCoVeR'].
        model_class (LightningModule): Class to load from checkpoint.
        device (str): 'cpu', 'cuda', or 'mps'.
        map_location: optional, for torch.load

    Returns:
        pd.DataFrame: Latents DataFrame including y if present.
    """
    model_path = Path(model_path)
    print(f"Loading {model_type} model from {model_path}...")

    # Load LightningModule from checkpoint and access the underlying model
    model = model_class.load_from_checkpoint(str(model_path), map_location=map_location).model
    model = model.to(device)
    model.eval()

    z_list, w_list, y_list = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
                y = batch[1].to(device) if len(batch) > 1 and torch.is_tensor(batch[1]) else None
            else:
                x, y = batch.to(device), None

            w = None

            if model_type == "Base":
                mu_z, logvar_z = model.encode(x)
                z = model.reparameterize(mu_z, logvar_z)
            elif model_type == "CVAE":
                mu_z, logvar_z = model.encode(x, y)
                z = model.reparameterize(mu_z, logvar_z)
            elif model_type in ["CSVAE", "DISCoVeR"]:
                mu_z, logvar_z, mu_w, logvar_w = model.encode(x, y)
                z = model.reparameterize(mu_z, logvar_z)
                w = model.reparameterize(mu_w, logvar_w)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            z_list.append(z.cpu())
            if w is not None:
                w_list.append(w.cpu())
            if y is not None:
                y_list.append(y.cpu())

    # Build DataFrame
    data = {}
    z_tensor = torch.cat(z_list)
    for i in range(z_tensor.shape[1]):
        data[f"z{i}"] = z_tensor[:, i].numpy()

    if w_list:
        w_tensor = torch.cat(w_list)
        for i in range(w_tensor.shape[1]):
            data[f"w{i}"] = w_tensor[:, i].numpy()

    if y_list:
        y_tensor = torch.cat(y_list)
        for i in range(y_tensor.shape[1]):
            data[f"y{i}"] = y_tensor[:, i].numpy()

    df = pd.DataFrame(data)

    # Save CSV
    output_csv = f"latents/latents_{model_type}_{len(dataloader.dataset)}.csv"
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved latents (including y) to {output_csv}")

    return df
