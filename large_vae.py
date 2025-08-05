from fg_funcs import vae_loss
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import WandbLogger
from torch import nn, optim
from torch.utils.data import DataLoader
import pytorch_lightning as L
import torch


# Create a flexible VAE model with customizable encoder/decoder hidden layers
class FullVAE(L.LightningModule):
    def __init__(self, input_dim, latent_dim, encoder_hidden_dims, decoder_hidden_dims):
        super(FullVAE, self).__init__()
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.LayerNorm(h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.1))  # Add dropout for regularization
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim * 2))  # Output mu and log_var
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.1))  # Add dropout for regularization
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # Assuming input is normalized between 0 and 1
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
    
# Create data loader
class FingerprintDataset(torch.utils.data.Dataset):
    def __init__(self, data, fingerprint_col='fingerprint_array', fg_col='fg_array'):
        self.data = data
        self.fingerprint_col = fingerprint_col
        self.fg_col = fg_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fingerprint = torch.tensor(row[self.fingerprint_col], dtype=torch.float32)
        fg_vector = torch.tensor(row[self.fg_col], dtype=torch.float32)
        return fingerprint, fg_vector, idx
    
class FullVAETrainer(L.LightningModule):
    def __init__(self, input_dim, encoder_hidden_dim, decoder_hidden_dim, latent_dim, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.model = FullVAE(
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
    
def train_vae_model(dataset, input_dim, latent_dim, encoder_hidden_dims, decoder_hidden_dims, batch_size=64, learning_rate=1e-3, max_epochs=50):
    # Split dataset into train, validation, and test sets
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

    # Create datasets
    train_dataset = FingerprintDataset(train_data)
    val_dataset = FingerprintDataset(val_data)
    test_dataset = FingerprintDataset(test_data)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    model = FullVAETrainer(
        input_dim=input_dim,
        encoder_hidden_dim=encoder_hidden_dims,
        decoder_hidden_dim=decoder_hidden_dims,
        latent_dim=latent_dim,
        learning_rate=learning_rate
    )

    # Initialize Wandb logger
    wandb_logger = WandbLogger(project='fg_vae', log_model=True)

    # Train the model
    trainer = L.Trainer(max_epochs=max_epochs, val_check_interval=0.5, logger=wandb_logger)

    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    trainer.test(model, test_loader)

    return model

# Example usage
if __name__ == "__main__":
    full_dataset = pd.read_csv('chembl_35_fg_full.csv')

    # Convert fingerprint to numpy array
    full_dataset['fingerprint_array'] = full_dataset['fingerprint_array'].apply(
        lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else np.array(x)
    )
    full_dataset['fg_array'] = full_dataset['fg_array'].apply(
        lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else np.array(x)
    )

    torch.manual_seed(42)  # For reproducibility

    input_dim = len(full_dataset['fingerprint_array'].iloc[0])  # Assuming fingerprint is a list of features
    latent_dim = 64  # Example latent dimension
    encoder_hidden_dims = [1024, 512, 256, 128]  # Example encoder hidden layers
    decoder_hidden_dims = [128, 256, 1024]  # Example decoder hidden layers

    vae_trainer = train_vae_model(
        dataset=full_dataset,
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=5
    )

    # Save the trained model
    torch.save(vae_trainer.state_dict(), 'full_vae_model.pth')
    print("VAE model trained and saved as 'full_vae_model.pth'")