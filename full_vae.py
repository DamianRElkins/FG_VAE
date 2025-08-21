import gc
from fg_funcs import vae_loss, save_model
import pandas as pd
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import WandbLogger
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as L
import torch
from scipy import sparse
import ast



# -------- Base MODEL --------
class BaseVAE(L.LightningModule):
    def __init__(self, input_dim, latent_dim, encoder_hidden_dims, decoder_hidden_dims):
        super(BaseVAE, self).__init__()
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
    
class FingerprintDataset(Dataset):
    def __init__(self, df):
        # Store the DataFrame directly. It now only contains sparse data columns.
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

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
        
        # You may not need the original index, but it's fine to keep it
        return fingerprint, fg_vector, idx
    

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
def train_base_model(dataset, input_dim, latent_dim, fg_dim, encoder_hidden_dims, decoder_hidden_dims, batch_size=64, learning_rate=1e-3, max_epochs=50):
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
    model = BaseVAETrainer(
        input_dim=input_dim,
        encoder_hidden_dim=encoder_hidden_dims,
        decoder_hidden_dim=decoder_hidden_dims,
        latent_dim=latent_dim,
        learning_rate=learning_rate
    )

    # Initialize Wandb logger
    wandb_logger = WandbLogger(project=f'fg_vae_{fg_dim}', log_model=True)

    # Train the model
    trainer = L.Trainer(max_epochs=max_epochs, val_check_interval=0.5, logger=wandb_logger)

    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    trainer.test(model, test_loader)

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
            encoder_layers.append(nn.LayerNorm(h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.1))
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim * 2))  # outputs mu and log_var
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: latent + FG vector
        decoder_layers = []
        prev_dim = latent_dim + fg_dim
        for h_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.1))
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
def train_conditional_vae(dataset, fingerprint_dim, fg_dim, latent_dim, encoder_hidden_dims, decoder_hidden_dims, batch_size=64, learning_rate=1e-3, max_epochs=50):
    # Split dataset
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.2, random_state=42)

    # Datasets & loaders
    train_dataset = FingerprintDataset(train_data)
    val_dataset = FingerprintDataset(val_data)
    test_dataset = FingerprintDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

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
    trainer = L.Trainer(max_epochs=max_epochs, val_check_interval=0.5, logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

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
            encoder_z_layers.append(nn.LayerNorm(h_dim))
            encoder_z_layers.append(nn.ReLU())
            encoder_z_layers.append(nn.Dropout(0.1))
            prev_dim = h_dim
        encoder_z_layers.append(nn.Linear(prev_dim, latent_dim_z * 2))
        self.encoder_z = nn.Sequential(*encoder_z_layers)

        # Encoder for latent space W (depends on x and y)
        encoder_w_layers = []
        prev_dim = fingerprint_dim + fg_dim
        for h_dim in encoder_hidden_dims_w:
            encoder_w_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_w_layers.append(nn.LayerNorm(h_dim))
            encoder_w_layers.append(nn.ReLU())
            encoder_w_layers.append(nn.Dropout(0.1))
            prev_dim = h_dim
        encoder_w_layers.append(nn.Linear(prev_dim, latent_dim_w * 2))
        self.encoder_w = nn.Sequential(*encoder_w_layers)

        # Decoder (depends on z and w)
        decoder_layers = []
        prev_dim = latent_dim_z + latent_dim_w
        for h_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.1))
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
        self.reconstruction_loss_func = nn.BCELoss(reduction='mean')
        self.adversarial_loss_func = nn.BCELoss(reduction='mean')

    def forward(self, x, fg):
        return self.model(x, fg)

    def training_step(self, batch, batch_idx):
        opt_vae, opt_adv = self.optimizers()
        x, fg, _ = batch

        recon_x, fg_pred, mu_z, log_var_z, mu_w, log_var_w = self(x, fg)
        z = self.model.reparameterize(mu_z, log_var_z)

        # ----- losses (use means) -----
        recon_loss = self.reconstruction_loss_func(recon_x, x)  # mean
        kld_z = -0.5 * (1 + log_var_z - mu_z.pow(2) - log_var_z.exp()).mean()
        kld_w = -0.5 * (1 + log_var_w - mu_w.pow(2) - log_var_w.exp()).mean()
        vae_loss_val = recon_loss + kld_z + kld_w

        # M2: maximize entropy => minimize (-H) for Bernoulli dims
        eps = 1e-10
        bernoulli_negH = (fg_pred * torch.log(fg_pred + eps) + (1 - fg_pred) * torch.log(1 - fg_pred + eps)).mean()
        m2_loss = bernoulli_negH  # this is -H

        total_vae_loss = self.beta1 * vae_loss_val + self.beta2 * m2_loss

        # ----- VAE step -----
        opt_vae.zero_grad()
        self.manual_backward(total_vae_loss) 
        opt_vae.step()

        # ----- adversary step (detach z) -----
        z_detached = z.detach()
        fg_pred_adv = self.model.classify_z(z_detached)
        n_loss = self.adversarial_loss_func(fg_pred_adv, fg)
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
        recon_loss = self.reconstruction_loss_func(recon_x, x)
        kld_z = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp())
        kld_w = -0.5 * torch.sum(1 + log_var_w - mu_w.pow(2) - log_var_w.exp())
        vae_loss_val = recon_loss + kld_z + kld_w
        n_loss = self.adversarial_loss_func(fg_pred, fg)
        self.log('val_vae_loss', vae_loss_val, on_epoch=True, prog_bar=True)
        self.log('val_adv_loss', n_loss, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', recon_loss, on_epoch=True)
        self.log('val_kld_z', kld_z, on_epoch=True)
        self.log('val_kld_w', kld_w, on_epoch=True)
        return vae_loss_val + n_loss
    
    def test_step(self, batch, batch_idx):
        x, fg, _ = batch
        recon_x, fg_pred, mu_z, log_var_z, mu_w, log_var_w = self(x, fg)
        recon_loss = self.reconstruction_loss_func(recon_x, x)
        kld_z = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp())
        kld_w = -0.5 * torch.sum(1 + log_var_w - mu_w.pow(2) - log_var_w.exp())
        vae_loss_val = recon_loss + kld_z + kld_w
        n_loss = self.adversarial_loss_func(fg_pred, fg)
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
def train_conditional_subspace_vae(dataset, fingerprint_dim, fg_dim, latent_dim_z, latent_dim_w, encoder_hidden_dims_z, encoder_hidden_dims_w, decoder_hidden_dims, adversarial_hidden_dims, batch_size=64, learning_rate=1e-3, max_epochs=50):
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.2, random_state=42)
    train_dataset = FingerprintDataset(train_data)
    val_dataset = FingerprintDataset(val_data)
    test_dataset = FingerprintDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
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
    trainer = L.Trainer(max_epochs=max_epochs, val_check_interval=0.5, logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
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
            encoder_z_layers.append(nn.LayerNorm(h_dim))
            encoder_z_layers.append(nn.ReLU())
            encoder_z_layers.append(nn.Dropout(0.1))
            prev_dim = h_dim
        encoder_z_layers.append(nn.Linear(prev_dim, latent_dim_z * 2))
        self.encoder_z = nn.Sequential(*encoder_z_layers)

        # Encoder for latent space W (depends on x and y)
        encoder_w_layers = []
        prev_dim = fingerprint_dim + fg_dim
        for h_dim in encoder_hidden_dims_w:
            encoder_w_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_w_layers.append(nn.LayerNorm(h_dim))
            encoder_w_layers.append(nn.ReLU())
            encoder_w_layers.append(nn.Dropout(0.1))
            prev_dim = h_dim
        encoder_w_layers.append(nn.Linear(prev_dim, latent_dim_w * 2))
        self.encoder_w = nn.Sequential(*encoder_w_layers)

        # Decoder 1 (full reconstruction, depends on z and w)
        decoder_layers = []
        prev_dim = latent_dim_z + latent_dim_w
        for h_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.1))
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
            decoder_z_layers.append(nn.Dropout(0.1))
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

        self.reconstruction_loss_func = nn.BCELoss(reduction='mean')
        self.adversarial_loss_func = nn.BCELoss(reduction='mean')

    def forward(self, x, fg):
        return self.model(x, fg)

    def training_step(self, batch, batch_idx):
        opt_vae, opt_adv = self.optimizers()
        x, fg, _ = batch

        recon_x_full, recon_x_hat, fg_pred, mu_z, log_var_z, mu_w, log_var_w = self(x, fg)

        # ----- VAE losses (main step) -----
        recon_loss_full = self.reconstruction_loss_func(recon_x_full, x)
        recon_loss_hat = self.reconstruction_loss_func(recon_x_hat, x)
        kld_z = -0.5 * (1 + log_var_z - mu_z.pow(2) - log_var_z.exp()).mean()
        kld_w = -0.5 * (1 + log_var_w - mu_w.pow(2) - log_var_w.exp()).mean()

        # M2: maximize entropy => minimize (-H) for Bernoulli dims
        neg_adv_loss = -self.adversarial_loss_func(fg_pred, fg)

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
        adv_loss_val = self.adversarial_loss_func(fg_pred_adv, fg)

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
        recon_loss_full = self.reconstruction_loss_func(recon_x_full, x)
        recon_loss_hat = self.reconstruction_loss_func(recon_x_hat, x)
        kld_z = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp())
        kld_w = -0.5 * torch.sum(1 + log_var_w - mu_w.pow(2) - log_var_w.exp())
        adv_loss_val = self.adversarial_loss_func(fg_pred, fg)
        total_loss = self.rec_w * recon_loss_full + self.kld_z_w * kld_z + self.kld_w_w * kld_w + self.rec_z_w * recon_loss_hat + self.adv_w * adv_loss_val

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
        recon_loss_full = self.reconstruction_loss_func(recon_x_full, x)
        recon_loss_hat = self.reconstruction_loss_func(recon_x_hat, x)
        kld_z = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp())
        kld_w = -0.5 * torch.sum(1 + log_var_w - mu_w.pow(2) - log_var_w.exp())
        adv_loss_val = self.adversarial_loss_func(fg_pred, fg)
        total_loss = self.rec_w * recon_loss_full + self.kld_z_w * kld_z + self.kld_w_w * kld_w + self.rec_z_w * recon_loss_hat + self.adv_w * adv_loss_val

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
def train_discover_vae(dataset, fingerprint_dim, fg_dim, latent_dim_z, latent_dim_w, encoder_hidden_dims_z, encoder_hidden_dims_w, decoder_hidden_dims, decoder_z_hidden_dims, adversarial_hidden_dims, batch_size=64, learning_rate=1e-3, max_epochs=50):
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.2, random_state=42)
    train_dataset = FingerprintDataset(train_data)
    val_dataset = FingerprintDataset(val_data)
    test_dataset = FingerprintDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
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
    trainer = L.Trainer(max_epochs=max_epochs, val_check_interval=0.5, logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    return model


if __name__ == "__main__":
    MODELS = ['DISCoVeR']
    MODEL_OUTPUT = 'models/large_models'

    # Load the raw data from CSV 
    full_dataset = pd.read_csv('data/chembl_35_fg_full.csv')
    
    torch.manual_seed(42)  # For reproducibility

    latent_dim = 16  # Example latent dimension
    encoder_hidden_dims = [1024, 512, 256, 128]  # Example encoder hidden layers
    decoder_hidden_dims = [128, 256, 1024]  # Example decoder hidden layers
    decoder_z_hidden_dims = [128, 256, 1024]  # Decoder layers for DISCoVeR
    latent_dim_z = 16 # for CSVAE
    latent_dim_w = 16 # for CSVAE
    encoder_hidden_dims_z = [1024, 512, 256, 128] # for CSVAE
    encoder_hidden_dims_w = [1024, 512, 256, 128] # for CSVAE
    adversarial_hidden_dims = [64] 
    batch_size = 64
    learning_rate = 1e-3
    max_epochs = 5

    fingerprint_dim = full_dataset['fp_length'].iloc[0]
    fg_dim = full_dataset['fg_length'].iloc[0]

    for MODEL in MODELS:
        if MODEL is None:
            raise ValueError("MODEL must be defined before training.")
        elif MODEL == 'Base':
            print("Training BaseVAE model...")

            vae_trainer = train_base_model(
                dataset=full_dataset,
                input_dim=fingerprint_dim,
                latent_dim=latent_dim,
                fg_dim=fg_dim,
                encoder_hidden_dims=encoder_hidden_dims,
                decoder_hidden_dims=decoder_hidden_dims,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_epochs=max_epochs
            )

            save_model(vae_trainer, MODEL_OUTPUT, MODEL)

        elif MODEL == 'CVAE':
            print("Training CVAE model...")

            vae_trainer = train_conditional_vae(
                dataset=full_dataset,
                fingerprint_dim=fingerprint_dim,
                fg_dim=fg_dim,
                latent_dim=latent_dim,
                encoder_hidden_dims=encoder_hidden_dims,
                decoder_hidden_dims=decoder_hidden_dims,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_epochs=max_epochs
            )

            save_model(vae_trainer, MODEL_OUTPUT, MODEL)

        elif MODEL == 'CSVAE':
                print("Training CSVAE model")

                vae_trainer = train_conditional_subspace_vae(
                    dataset=full_dataset,
                    fingerprint_dim=fingerprint_dim,
                    fg_dim=fg_dim,
                    latent_dim_z=latent_dim_z,
                    latent_dim_w=latent_dim_w,
                    encoder_hidden_dims_z=encoder_hidden_dims_z,
                    encoder_hidden_dims_w=encoder_hidden_dims_w,
                    decoder_hidden_dims=decoder_hidden_dims,
                    adversarial_hidden_dims=adversarial_hidden_dims,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs
                )
                save_model(vae_trainer, MODEL_OUTPUT, MODEL)

        elif MODEL == 'DISCoVeR':
                print("Training DISCoVeR VAE model...")

                vae_trainer = train_discover_vae(
                    dataset=full_dataset,
                    fingerprint_dim=fingerprint_dim,
                    fg_dim=fg_dim,
                    latent_dim_z=latent_dim_z,
                    latent_dim_w=latent_dim_w,
                    encoder_hidden_dims_z=encoder_hidden_dims_z,
                    encoder_hidden_dims_w=encoder_hidden_dims_w,
                    decoder_hidden_dims=decoder_hidden_dims,
                    decoder_z_hidden_dims=decoder_z_hidden_dims,
                    adversarial_hidden_dims=adversarial_hidden_dims,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs
                )
                save_model(vae_trainer, MODEL_OUTPUT, MODEL)

        # Free memory after each model
        del vae_trainer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
