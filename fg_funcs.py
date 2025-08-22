import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import os
import pickle
import wandb
import efgs
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import pandas as pd
from rdkit import DataStructs
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import WandbLogger
from torch import optim
import pytorch_lightning as L
from scipy import sparse
import ast
from pathlib import Path

PATT: Chem.Mol = Chem.MolFromSmarts("[$([D1]=[*])]")
REPL: Chem.Mol = Chem.MolFromSmiles("*")

def get_scaffold(
    mol: Chem.Mol,
    real_bm: bool = True,
    use_csk: bool = False,
    use_bajorath: bool = False
) -> Chem.Mol:
    """
    Generate the scaffold of a molecule using MurckoScaffold, with options for Bajorath and CSK modifications.
    """
    Chem.RemoveStereochemistry(mol)
    scaff = MurckoScaffold.GetScaffoldForMol(mol)
    if use_bajorath:
        scaff = AllChem.DeleteSubstructs(scaff, PATT)
    if real_bm:
        scaff = AllChem.ReplaceSubstructs(scaff, PATT, REPL, replaceAll=True)[0]
    if use_csk:
        scaff = MurckoScaffold.MakeScaffoldGeneric(scaff)
        if real_bm:
            scaff = MurckoScaffold.GetScaffoldForMol(scaff)
    return scaff

def save_chunk_results(
    dir: str,
    chunk_idx: int,
    results: Any
) -> None:
    """
    Save results for a data chunk to a pickle file in the specified directory.
    """
    with open(os.path.join(dir, f"chunk_{chunk_idx}.pkl"), "wb") as f:
        pickle.dump(results, f)

def safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """
    Safely convert a SMILES string to an RDKit Mol object, returning None if conversion fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None

def compute_efgs(mol: Chem.Mol) -> List[str]:
    """
    Compute extended functional groups (EFGs) for a molecule.
    """
    _, _, psmis, _ = efgs.get_dec_fgs(mol)
    return psmis

def compute_efgs_safe(mol: Chem.Mol) -> Optional[List[str]]:
    """
    Safely compute EFGs for a molecule, returning None if computation fails.
    """
    try:
        return compute_efgs(mol)
    except Exception as e:
        print(f"Error computing EFGs for molecule: {mol}. Error: {e}")
        return None

def compute_scaffold_safe(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Safely compute the scaffold for a molecule, returning None if computation fails.
    """
    try:
        return get_scaffold(mol)
    except Exception:
        return None

def create_mfpgen() :
    """
    Create a Morgan fingerprint generator with radius 2 and 2048 bits.
    """
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

mfpgen = create_mfpgen()

def mol_to_fingerprint(mol: Optional[Chem.Mol]) -> Optional[DataStructs.cDataStructs.ExplicitBitVect]:
    """
    Convert an RDKit Mol object to a Morgan fingerprint bit vector.
    """
    if mol is None:
        return None
    return mfpgen.GetFingerprint(mol)

def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Convert a SMILES string to an RDKit Mol object, returning None if conversion fails.
    """
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception as e:
        print(f"Error converting SMILES {smiles}: {e}")
        return None

def fg_to_array(fgs: Union[List[str], None], fg_list: List[str]) -> np.ndarray:
    """
    Convert a list of functional group SMILES to a binary array indicating presence in fg_list.
    """
    fg_array = np.zeros(len(fg_list), dtype=int)
    if isinstance(fgs, list):
        for fg in fgs:
            if fg in fg_list:
                fg_array[fg_list.index(fg)] = 1
    return fg_array

def fp_to_array(fp: DataStructs.cDataStructs.ExplicitBitVect) -> np.ndarray:
    """
    Convert an RDKit fingerprint bit vector to a numpy array.
    """
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

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
    bce = nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (bce + beta * kld) , bce , kld 

def visualize_latent_space(
    model: nn.Module,
    dataset: Dataset,
    fg_indicies: List[int] = [0, 1, 2, 3],
    method: str = 'tsne',
    sample_size: Optional[int] = None
) -> None:
    """
    Visualize the latent space of a VAE model using t-SNE, PCA, or UMAP, colored by functional group combinations.
    """
    model.eval()
    latents: List[torch.Tensor] = []
    fg_labels: List[torch.Tensor] = []

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            x, fg_vecs, _ = batch
            _, mu, _ = model(x)
            latents.append(mu.cpu())
            fg_labels.append(fg_vecs.cpu())

    latents_np = torch.cat(latents).numpy()
    fg_labels_np = torch.cat(fg_labels).numpy()

    if sample_size and sample_size < len(latents_np):
        indices = np.random.choice(len(latents_np), sample_size, replace=False)
        latents_np = latents_np[indices]
        fg_labels_np = fg_labels_np[indices]

    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'umap':
        reducer = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid method. Choose from 'tsne', 'pca', 'umap'.")

    latents_2d = reducer.fit_transform(latents_np)

    combined_labels: List[str] = []
    for i in range(len(fg_labels_np)):
        label_parts: List[str] = []
        for fg_idx in fg_indicies:
            label_parts.append(f"{int(fg_labels_np[i, fg_idx])}")
        combined_labels.append("_".join(label_parts))

    plt.figure(figsize=(8, 7))
    sns.scatterplot(x=latents_2d[:, 0], y=latents_2d[:, 1], hue=combined_labels,
                    palette='tab10', alpha=0.8, s=20)

    fg_indices_str = ', '.join(map(str, fg_indicies))
    plt.title(f"Latent space (colored by functional groups [{fg_indices_str}]) Using {method.upper()}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(title="Functional Group Combinations", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

def perform_umap_on_fingerprints(
    data: Any,
    fingerprint_col: str = 'fingerprint_array',
    fg_col: str = 'fg_array',
    fg_indicies: List[int] = [0, 1, 2, 3],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Perform UMAP dimensionality reduction on molecular fingerprints and visualize colored by functional groups.
    Returns the UMAP embeddings.
    """
    fingerprints = np.array(data[fingerprint_col].tolist())
    fg_labels = np.array(data[fg_col].tolist())

    umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    umap_embeddings = umap_model.fit_transform(fingerprints)

    combined_labels: List[str] = []
    for i in range(len(fg_labels)):
        label_parts: List[str] = []
        for fg_idx in fg_indicies:
            if fg_idx < fg_labels.shape[1]:
                label_parts.append(f"FG{fg_idx}_{int(fg_labels[i, fg_idx])}")
            else:
                label_parts.append(f"FG{fg_idx}_N/A")
        combined_labels.append("_".join(label_parts))

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1], hue=combined_labels,
                    palette='tab10', alpha=0.8, s=20)

    fg_indices_str = ', '.join(map(str, fg_indicies))
    plt.title(f"UMAP of Fingerprints Colored by Functional Groups [{fg_indices_str}]")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Functional Group Combinations", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

    return umap_embeddings

def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute F1 score, precision, and recall for binary classification arrays.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fp == 0 or tp + fn == 0:
        return 0.0, 0.0, 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0, 0.0, 0.0

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1, precision, recall

def matthews_corrcoef(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute the Matthews correlation coefficient for binary classification arrays.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return numerator / denominator

def calculate_reconstruction_quality(
    model: nn.Module,
    dataset: Dataset,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Calculate reconstruction quality metrics for a VAE model, including Tanimoto, F1, precision, recall, uniqueness, and MCC.
    """
    model.eval()
    all_fingerprints: List[torch.Tensor] = []
    all_reconstructions: List[torch.Tensor] = []
    all_fg_labels: List[torch.Tensor] = []

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            x, fg_vecs, _ = batch
            recon_x, _, _ = model(x)
            all_fingerprints.append(x.cpu())
            all_reconstructions.append(recon_x.cpu())
            all_fg_labels.append(fg_vecs.cpu())

    all_fingerprints_np = torch.cat(all_fingerprints).numpy()
    all_reconstructions_np = torch.cat(all_reconstructions).numpy()

    tanimoto_scores: List[float] = []
    for i in range(len(all_fingerprints_np)):
        fp1 = Chem.DataStructs.cDataStructs.CreateFromBitString(''.join(map(str, all_fingerprints_np[i].astype(int))))
        fp2 = Chem.DataStructs.cDataStructs.CreateFromBitString(''.join(map(str, (all_reconstructions_np[i] > threshold).astype(int))))
        tanimoto_scores.append(DataStructs.TanimotoSimilarity(fp1, fp2))

    tanimoto_scores_np = np.array(tanimoto_scores)

    f1_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    mcc_scores: List[float] = []
    for i in range(len(all_fingerprints_np)):
        y_true = all_fingerprints_np[i]
        y_pred = all_reconstructions_np[i] > threshold
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        f1, precision, recall = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        mcc_scores.append(mcc)

    f1_scores_np = np.array(f1_scores)
    precision_scores_np = np.array(precision_scores)
    recall_scores_np = np.array(recall_scores)
    mcc_scores_np = np.array(mcc_scores)

    unique_reconstructions = np.unique(all_reconstructions_np[all_reconstructions_np > threshold], axis=0)
    uniqueness = len(unique_reconstructions) / len(all_reconstructions_np[all_reconstructions_np > threshold])

    return tanimoto_scores_np, f1_scores_np, precision_scores_np, recall_scores_np, uniqueness, mcc_scores_np

def average_latent_vector(
    model: nn.Module,
    dataset: Dataset,
    fg_col: str = 'fg_array',
    fg_indicies: List[int] = [0, 1, 2, 3]
) -> Dict[int, Optional[np.ndarray]]:
    """
    Compute the average latent vector for each functional group index in the dataset.
    """
    model.eval()
    fg_latents: Dict[int, List[np.ndarray]] = {fg_idx: [] for fg_idx in fg_indicies}

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            x, fg_vecs, _ = batch
            _, mu, _ = model(x)
            mu_np = mu.cpu().numpy()
            fg_vecs_np = fg_vecs.cpu().numpy()

            for i in range(len(mu_np)):
                for fg_idx in fg_indicies:
                    if fg_vecs_np[i, fg_idx] == 1:
                        fg_latents[fg_idx].append(mu_np[i])

    avg_latents: Dict[int, Optional[np.ndarray]] = {
        fg_idx: np.mean(fg_latents[fg_idx], axis=0) if fg_latents[fg_idx] else None
        for fg_idx in fg_indicies
    }
    return avg_latents

def plot_average_latent_vectors(
    avg_latents: Dict[int, Optional[np.ndarray]]
) -> None:
    """
    Plot boxplots of the average latent vectors for each functional group across latent dimensions.
    """
    if not avg_latents:
        print("No data to plot.")
        return

    latent_dim = len(next(iter(avg_latents.values())))

    data_for_boxplot: List[List[float]] = [[] for _ in range(latent_dim)]

    for fg_index in sorted(avg_latents.keys()):
        latent_vector = avg_latents[fg_index]
        if latent_vector is None:
            latent_vector = np.zeros(latent_dim)

        for i in range(latent_dim):
            data_for_boxplot[i].append(latent_vector[i])

    plt.figure(figsize=(12, 6))
    plt.boxplot(data_for_boxplot)

    plt.title('Distribution of Average Latent Values Across Functional Groups per Latent Dimension')
    plt.xlabel('Latent Dimension Index')
    plt.ylabel('Average Latent Value')
    plt.xticks(np.arange(1, latent_dim + 1), range(latent_dim))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_distributions(
    plot_info_list: List[Dict[str, Any]],
    bins: int = 50,
    figsize_per_plot: Tuple[int, int] = (5, 4)
) -> None:
    """
    Plot multiple distributions as histograms with KDE, using provided plot information.
    """
    num_plots = len(plot_info_list)

    if num_plots == 0:
        print("No plot information provided.")
        return

    ncols = min(num_plots, 3)
    nrows = math.ceil(num_plots / ncols)

    total_figsize_width = ncols * figsize_per_plot[0]
    total_figsize_height = nrows * figsize_per_plot[1]

    fig, axes = plt.subplots(nrows, ncols, figsize=(total_figsize_width, total_figsize_height))

    if num_plots == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, plot_info in enumerate(plot_info_list):
        ax = axes[i]
        data = plot_info['data']
        title = plot_info['title']
        xlabel = plot_info['xlabel']

        sns.histplot(data, bins=bins, kde=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')

    for j in range(num_plots, nrows * ncols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def get_nearest_neighbors(
    model: nn.Module,
    dataset: Dataset,
    latents: Optional[np.ndarray] = None,
    n_neighbors: int = 50
) -> Tuple[NearestNeighbors, np.ndarray, np.ndarray]:
    """
    Fit a NearestNeighbors model to the latent vectors of a dataset, optionally using provided latents.
    Returns the NearestNeighbors model, latent vectors, and functional group labels.
    """
    model.eval()
    if latents is None:
        latents_list: List[torch.Tensor] = []
        fg_labels_list: List[torch.Tensor] = []
        with torch.no_grad():
            for x, fg, _ in DataLoader(dataset, batch_size=64, shuffle=False):
                _, mu, _ = model(x)
                latents_list.append(mu.cpu())
                fg_labels_list.append(fg.cpu())
        latents = torch.cat(latents_list).numpy()
        fg_labels = torch.cat(fg_labels_list).numpy()
    else:
        fg_labels = np.array([])

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(latents)
    return nbrs, latents, fg_labels

def find_nearest_neighbors(
    nbrs: NearestNeighbors,
    query_latent: np.ndarray,
    n_neighbors: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the nearest neighbors for a query latent vector using a fitted NearestNeighbors model.
    Returns distances and indices of neighbors.
    """
    distances, indices = nbrs.kneighbors(query_latent.reshape(1, -1), n_neighbors=n_neighbors)
    return distances.flatten(), indices.flatten()

def get_fg_counts(
    dataset: Dataset,
    fg_col: str = 'fg_array'
) -> np.ndarray:
    """
    Calculate the normalized counts of each functional group in the dataset.
    """
    df = dataset.data
    fg_list = df[fg_col].tolist()
    fg_counts: Dict[int, int] = {i: 0 for i in range(len(fg_list[0]))}
    for fg in fg_list:
        for i, val in enumerate(fg):
            if val == 1:
                fg_counts[i] += 1
    fg_counts_arr = np.array(list(fg_counts.values()), dtype=float)
    return fg_counts_arr / len(fg_list)

def weighted_tanimoto(
    fg_a: np.ndarray,
    fg_b: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    Compute the weighted Tanimoto similarity between two binary arrays, using provided weights.
    """
    intersection = np.sum(weights * (fg_a * fg_b))
    union = np.sum(weights * ((fg_a + fg_b) > 0))
    return intersection / union if union > 0 else 0.0

def metric(
    nbrs: NearestNeighbors,
    dataset: Dataset,
    query_index: int,
    latent: np.ndarray,
    fg_counts: np.ndarray,
    fg_col: str = 'fg_array',
    n_neighbors: int = 50
) -> float:
    """
    Combined metric: Weighted Tanimoto similarity normalized by expected random similarity.
    """
    query_fg = dataset.data.iloc[query_index][fg_col]  # np.ndarray
    distances, indices = find_nearest_neighbors(nbrs, latent[query_index], n_neighbors)

    # Inverse prevalence weights (avoid division by zero)
    weights = 1.0 / (fg_counts + 1e-8)
    weights /= np.sum(weights)

    tanimoto_scores = np.zeros(len(indices), dtype=float)
    for i, idx in enumerate(indices):
        neighbor_fg = dataset.data.iloc[idx][fg_col]
        tanimoto_scores[i] = weighted_tanimoto(query_fg, neighbor_fg, weights)

    avg_weighted_tanimoto = np.mean(tanimoto_scores) if len(tanimoto_scores) > 0 else 0.0

    return avg_weighted_tanimoto #  [1,0,0,1]

def save_model(model, output_dir, model_name):
    """Saves a given model to output directory"""
    # Save the trained model
    if not os.path.exists(f'{output_dir}'):
        os.makedirs(f'{output_dir}')
    torch.save(model.state_dict(), f'{output_dir}/{model_name}_vae_model.pth')
    print(f"{model_name} model trained and saved as '{output_dir}/{model_name}_vae_model.pth'")

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

    # Train the model
    trainer = L.Trainer(max_epochs=max_epochs, val_check_interval=0.5, logger=wandb_logger)

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
    trainer = L.Trainer(max_epochs=max_epochs, val_check_interval=0.5, logger=wandb_logger)
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
    trainer = L.Trainer(max_epochs=max_epochs, val_check_interval=0.5, logger=wandb_logger)
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
    trainer = L.Trainer(max_epochs=max_epochs, val_check_interval=0.5, logger=wandb_logger)
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
