import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import os
import pickle
import efgs
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
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