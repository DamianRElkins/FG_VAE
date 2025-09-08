import os
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score

from scipy.stats import gaussian_kde

from umap import UMAP

from rdkit import DataStructs
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.DataStructs import ExplicitBitVect

import efgs
import colorsys

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

def fingerprint_to_bv(fp: np.ndarray) -> ExplicitBitVect:
    """Convert a numpy fingerprint array to RDKit ExplicitBitVect."""
    bv = ExplicitBitVect(len(fp))
    for i, bit in enumerate(fp.astype(int)):
        if bit:
            bv.SetBit(i)
    return bv

def visualize_latent_space_tanimoto(
    model: nn.Module,
    dataset: Dataset,
    method: str = 'tsne',
    sample_size: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    use_tanimoto_brightness: bool = True
) -> None:
    """
    Visualize the latent space of a VAE model using t-SNE, PCA, or UMAP,
    colored by the most frequent active FG, with optional brightness based on
    Tanimoto similarity to the most central point.
    """
    model.eval()
    latents, fg_labels, fingerprints = [], [], []

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            x, fg_vecs, _ = batch  # x is fingerprint or input tensor
            _, mu, _ = model(x)
            latents.append(mu.cpu())
            fg_labels.append(fg_vecs.cpu())
            fingerprints.append(x.cpu())

    latents = torch.cat(latents).numpy()
    fg_labels = torch.cat(fg_labels).numpy()
    fingerprints = torch.cat(fingerprints).numpy()

    np.random.seed(42)

    if sample_size and sample_size < len(latents):
        indices = np.random.choice(len(latents), sample_size, replace=False)
        latents = latents[indices]
        fg_labels = fg_labels[indices]
        fingerprints = fingerprints[indices]

    # Compute FG frequencies
    fg_counts = np.sum(fg_labels, axis=0)

    # Assign each point to the most frequent FG among its actives
    point_fgs = []
    for row in fg_labels:
        active_fgs = np.where(row == 1)[0]
        if len(active_fgs) == 0:
            point_fgs.append(-1)
        else:
            point_fgs.append(active_fgs[np.argmax(fg_counts[active_fgs])])
    point_fgs = np.array(point_fgs)

    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'umap':
        reducer = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid method. Choose from 'tsne', 'pca', 'umap'.")

    latents_2d = reducer.fit_transform(latents)

    # ----- Compute Tanimoto-based brightness -----
    brightness = None
    if use_tanimoto_brightness:
        center_idx = np.argmin(np.linalg.norm(latents_2d - latents_2d.mean(axis=0), axis=1))
        center_fp = fingerprint_to_bv(fingerprints[center_idx])

        # Compute similarities for all points
        similarities = np.array([
            TanimotoSimilarity(fingerprint_to_bv(fp), center_fp)
            for fp in fingerprints
        ])

        # Exclude center point from normalization
        mask = np.ones_like(similarities, dtype=bool)
        mask[center_idx] = False
        sim_for_norm = similarities[mask]

        # Print statistics excluding center
        print(f"Similarity SD: {sim_for_norm.std():.4f}, Mean: {sim_for_norm.mean():.4f}")

        # Standard normalization
        norm = (sim_for_norm - sim_for_norm.mean()) / sim_for_norm.std()

        # Exponential scaling to exaggerate differences
        norm = np.exp(norm)  # all positive, larger values more distinct

        # Normalize to [0.2, 1.0] for V
        norm = 0.2 + 0.8 * (norm - norm.min()) / (norm.max() - norm.min())


        # Reinsert brightness with center point forced to full brightness (1.0)
        brightness = np.empty_like(similarities)
        brightness[mask] = norm
        brightness[center_idx] = 1.0

    # ----- Plot -----
    plt.figure(figsize=(8, 7), dpi=500)

    # Get colors for each FG
    palette = sns.color_palette('tab10', n_colors=len(np.unique(point_fgs)))
    fg_to_color = {fg: palette[i % len(palette)] for i, fg in enumerate(np.unique(point_fgs))}
    base_colors = np.array([fg_to_color[fg] for fg in point_fgs])

    if brightness is not None:
        # Convert RGB to HSV, adjust V, then back to RGB
        colors = []
        for rgb, v in zip(base_colors, brightness):
            hsv = colorsys.rgb_to_hsv(*rgb)
            rgb_new = colorsys.hsv_to_rgb(hsv[0], hsv[1], v)
            colors.append(rgb_new)
        colors = np.array(colors)
    else:
        colors = base_colors

    # Plot all points except the center
    mask = np.ones(len(latents_2d), dtype=bool)
    mask[center_idx] = False
    plt.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                c=colors[mask], s=20, alpha=0.8)

    # Plot the center point last with a dark circle
    plt.scatter(latents_2d[center_idx, 0], latents_2d[center_idx, 1],
                c=[colors[center_idx]], s=60, edgecolor='k', linewidth=1.5, zorder=10)

    plt.title(title or f"Latent Space Colored by Most Frequent FG ({method.upper()})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.show()



def visualize_latent_space_3d_kde(
    model: nn.Module,
    dataset: Dataset,
    method: str = 'tsne',
    sample_size: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    use_tanimoto_z: bool = True
):
    """
    Visualize the latent space of a VAE model in 3D using t-SNE, PCA, or UMAP,
    colored by most frequent FG, with Z-axis optionally representing Tanimoto similarity
    and an overlaid KDE surface to show density.
    """
    model.eval()
    latents, fg_labels, fingerprints = [], [], []

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            x, fg_vecs, _ = batch
            _, mu, _ = model(x)
            latents.append(mu.cpu())
            fg_labels.append(fg_vecs.cpu())
            fingerprints.append(x.cpu())

    latents = torch.cat(latents).numpy()
    fg_labels = torch.cat(fg_labels).numpy()
    fingerprints = torch.cat(fingerprints).numpy()

    if sample_size and sample_size < len(latents):
        indices = np.random.choice(len(latents), sample_size, replace=False)
        latents = latents[indices]
        fg_labels = fg_labels[indices]
        fingerprints = fingerprints[indices]

    # Assign dominant FG per point
    fg_counts = np.sum(fg_labels, axis=0)
    point_fgs = []
    for row in fg_labels:
        active_fgs = np.where(row == 1)[0]
        point_fgs.append(active_fgs[np.argmax(fg_counts[active_fgs])] if len(active_fgs) > 0 else -1)
    point_fgs = np.array(point_fgs)

    # Dimensionality reduction
    reducer = {'tsne': TSNE, 'pca': PCA, 'umap': UMAP}[method](n_components=2, random_state=42)
    latents_2d = reducer.fit_transform(latents)

    # Compute Z-axis (Tanimoto similarity)
    if use_tanimoto_z:
        center_idx = np.argmin(np.linalg.norm(latents_2d - latents_2d.mean(axis=0), axis=1))
        center_fp = fingerprint_to_bv(fingerprints[center_idx])
        z = np.array([TanimotoSimilarity(fingerprint_to_bv(fp), center_fp) for fp in fingerprints])
    else:
        z = np.zeros(len(latents_2d))

    # ----- Compute 3D KDE for density -----
    xyz = np.vstack([latents_2d[:, 0], latents_2d[:, 1], z])
    kde = gaussian_kde(xyz)(xyz)
    kde_norm = (kde - kde.min()) / (kde.max() - kde.min())  # normalize to [0,1]

    # ----- 3D Scatter Plot -----
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    palette = sns.color_palette('tab10', n_colors=len(np.unique(point_fgs)))
    fg_to_color = {fg: palette[i % len(palette)] for i, fg in enumerate(np.unique(point_fgs))}
    colors = [fg_to_color[fg] for fg in point_fgs]

    # Scatter plot colored by FG, size scaled by KDE density
    ax.scatter(
        latents_2d[:, 0], latents_2d[:, 1], z,
        c=colors, s=20 + 100*kde_norm, alpha=0.6
    )

    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    ax.set_zlabel("Tanimoto Similarity" if use_tanimoto_z else "Z")
    ax.set_title(title or f"3D Latent Space with KDE ({method.upper()})")
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "latent_space_3d_kde.png"), dpi=300)
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

def visualize_latent_space_per_fg(
    latents_list: List[np.ndarray],
    fg_list: List[np.ndarray],
    method: str = 'tsne',
    sample_size: Optional[int] = None,
    combined_title: Optional[str] = None,
    per_fg_title: Optional[str] = None,
    save_path: Optional[str] = None,
    fg_names: List[str] = None
) -> None:
    """
    Visualizes a latent space with two plots:
    1. Combined scatter plot labeled by the most frequent FG among active ones.
    2. Grid of subplots, each showing one FG (multiclass points appear in all their active FGs).

    Parameters:
        latents_list (List[np.ndarray]): List of latent arrays (N_i x latent_dim).
        fg_list (List[np.ndarray]): List of binary arrays (N_i x num_FGs) corresponding to each latent array.
        method (str): Dimensionality reduction method ('tsne', 'pca', 'umap').
        sample_size (Optional[int]): Number of points to sample for visualization.
        combined_title (Optional[str]): Title for the combined plot.
        per_fg_title (Optional[str]): Title for the per-FG subplot figure.
        save_path (Optional[str]): Directory path to save the figures. If None, figures are not saved.
        fg_names (List[str]): List of molecule names, one per point (must match total N).
    """
    # Concatenate all latents and FG arrays
    latents = np.vstack(latents_list)
    fg_labels = np.vstack(fg_list)  # Shape: (N, num_fgs)

    np.random.seed(42)

    if sample_size and sample_size < len(latents):
        indices = np.random.choice(len(latents), sample_size, replace=False)
        latents = latents[indices]
        fg_labels = fg_labels[indices]
        if fg_names is not None:
            fg_names = [fg_names[i] for i in indices]

    # Compute global frequencies of each FG
    fg_counts = np.sum(fg_labels, axis=0)

    # Assign each point to the most frequent FG among its active ones
    point_fgs = []
    for row in fg_labels:
        active_fgs = np.where(row == 1)[0]
        if len(active_fgs) == 0:
            point_fgs.append(-1)  # no active FG
        else:
            active_freqs = fg_counts[active_fgs]
            point_fgs.append(active_fgs[np.argmax(active_freqs)])
    point_fgs = np.array(point_fgs)

    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'umap':
        reducer = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid method. Choose from 'tsne', 'pca', 'umap'.")

    latents_2d = reducer.fit_transform(latents)

    # ----- Plot 1: Combined -----
    plt.figure(figsize=(8, 7))
    # Map FG indices to names if fg_names is provided
    if fg_names is not None and isinstance(fg_names, list) and len(fg_names) == fg_labels.shape[1]:
        unique_fgs = np.unique(point_fgs)
        sorted_fgs = sorted(unique_fgs)
        fg_labels_map = {fg: (fg_names[fg] if fg >= 0 and fg < len(fg_names) else "None") for fg in sorted_fgs}
        hue_labels = [fg_labels_map[fg] for fg in point_fgs]
        palette_colors = sns.color_palette('tab10', n_colors=len(sorted_fgs))
        palette = dict(zip([fg_labels_map[fg] for fg in sorted_fgs], palette_colors))
        scatter = sns.scatterplot(x=latents_2d[:, 0], y=latents_2d[:, 1], hue=hue_labels,
                                  palette=palette, alpha=0.8, s=20)
        handles, labels = scatter.get_legend_handles_labels()
        label_order = [fg_labels_map[fg] for fg in sorted_fgs]
        ordered_handles = [handles[labels.index(lbl)] for lbl in label_order if lbl in labels]
        scatter.legend(ordered_handles, label_order, title="Functional Groups",
                       bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        fg_idx_to_color = {fg: palette[fg_labels_map[fg]] for fg in sorted_fgs}
        fg_idx_to_name = {fg: fg_labels_map[fg] for fg in sorted_fgs}
        fg_idx_to_color[-1] = (0.5, 0.5, 0.5)
        fg_idx_to_name[-1] = "None"
    else:
        unique_fgs = np.unique(point_fgs)
        sorted_fgs = sorted(unique_fgs)
        palette_colors = sns.color_palette('tab10', n_colors=len(sorted_fgs))
        fg_idx_to_color = {fg: palette_colors[i] for i, fg in enumerate(sorted_fgs)}
        fg_idx_to_name = {fg: str(fg) for fg in sorted_fgs}
        fg_idx_to_color[-1] = (0.5, 0.5, 0.5)
        fg_idx_to_name[-1] = "None"
        scatter = sns.scatterplot(x=latents_2d[:, 0], y=latents_2d[:, 1], hue=point_fgs,
                                  palette=fg_idx_to_color, alpha=0.8, s=20)
        handles, labels = scatter.get_legend_handles_labels()
        label_order = [str(fg) for fg in sorted_fgs]
        ordered_handles = [handles[labels.index(lbl)] for lbl in label_order if lbl in labels]
        scatter.legend(ordered_handles, label_order, title="Functional Groups",
                       bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.title(combined_title or f"Latent space colored by MOST FREQUENT FG ({method.upper()})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "combined_latent_space.png"), dpi=300)
    plt.show()

    # ----- Plot 2: One subplot per FG -----
    num_fgs = fg_labels.shape[1]
    cols = min(4, num_fgs)
    rows = int(np.ceil(num_fgs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    # Use fg_names for molecule names if provided and length matches number of points
    molecule_names = fg_names if fg_names is not None and len(fg_names) == len(latents_2d) else None

    for idx in range(num_fgs):
        mask = fg_labels[:, idx] == 1  # includes multiclass points
        multiclass_mask = (fg_labels.sum(axis=1) > 1) & mask
        singleclass_mask = (fg_labels.sum(axis=1) == 1) & mask

        # Plot single-class points in their original color
        singleclass_fgs = point_fgs[singleclass_mask]
        singleclass_colors = [fg_idx_to_color[fg] for fg in singleclass_fgs]
        axes[idx].scatter(latents_2d[singleclass_mask, 0], latents_2d[singleclass_mask, 1],
                          alpha=0.8, s=20, color=singleclass_colors, label="Single FG")

        # For multiclass points, use the color of their alternate class (not idx)
        multiclass_indices = np.where(multiclass_mask)[0]
        for i in multiclass_indices:
            # Find alternate class (not idx) among active FGs
            active_fgs = np.where(fg_labels[i] == 1)[0]
            alternate_fgs = [fg for fg in active_fgs if fg != idx]
            # Use the color of the alternate class with highest frequency (or just the first)
            if alternate_fgs:
                alt_fg = alternate_fgs[np.argmax(fg_counts[alternate_fgs])]
                color = fg_idx_to_color.get(alt_fg, "red")
            else:
                color = "red"
            axes[idx].scatter(latents_2d[i, 0], latents_2d[i, 1],
                              alpha=0.8, s=20, color=[color], label="Multi FG" if i == multiclass_indices[0] else None)
            # Annotate with molecule name if available
            if molecule_names is not None:
                axes[idx].annotate(str(molecule_names[i]), (latents_2d[i, 0], latents_2d[i, 1]),
                                   fontsize=7, alpha=0.7, xytext=(2, 2), textcoords='offset points')

        # Optionally annotate singleclass points as well
        if molecule_names is not None:
            for i in np.where(singleclass_mask)[0]:
                axes[idx].annotate(str(molecule_names[i]), (latents_2d[i, 0], latents_2d[i, 1]),
                                   fontsize=7, alpha=0.7, xytext=(2, 2), textcoords='offset points')

        axes[idx].set_title(f"{fg_idx_to_name.get(idx, f'FG {idx}')} (n={mask.sum()})")
        axes[idx].set_xlabel("Dim 1")
        axes[idx].set_ylabel("Dim 2")
        # Only show unique legend entries
        handles, labels = axes[idx].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[idx].legend(by_label.values(), by_label.keys(), loc="best", fontsize="small")

    # Hide unused subplots
    for ax in axes[num_fgs:]:
        ax.axis('off')

    fig.suptitle(per_fg_title or f"Latent space split by each FG ({method.upper()})")
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, "per_fg_latent_space.png"), dpi=300)
    plt.show()


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
    model: Optional[nn.Module] = None,
    dataset: Optional[Dataset] = None,
    latents: Optional[np.ndarray] = None,
    fg_labels: Optional[np.ndarray] = None,
    n_neighbors: int = 50
) -> Tuple[NearestNeighbors, np.ndarray, np.ndarray]:
    """
    Fit a NearestNeighbors model to the latent vectors of a dataset, optionally using provided latents.
    Returns the NearestNeighbors model, latent vectors, and functional group labels. Latents must be passed for
    CVAE and DISCoVeR
    """
    if model is not None:
        model.eval()
    if latents is None:
        if model is None:
            raise ValueError("Model must be provided if latents are not available.")
        latents_list: List[torch.Tensor] = []
        fg_labels_list: List[torch.Tensor] = []
        with torch.no_grad():
            for x, fg, _ in DataLoader(dataset, batch_size=64, shuffle=False):
                _, mu, _ = model(x)
                latents_list.append(mu.cpu())
                fg_labels_list.append(fg.cpu())
        latents = torch.cat(latents_list).numpy()
        fg_labels = torch.cat(fg_labels_list).numpy()
    elif fg_labels is None:
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
    df = dataset.df
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
    query_index: int,
    latent: np.ndarray,
    fg_counts: np.ndarray,
    fg_labels: np.ndarray,
    n_neighbors: int = 50,
    n_random: int = 100
) -> float:
    """
    Combined metric: Weighted Tanimoto similarity normalized by expected random similarity.
    """
    # Step 1: Find neighbors for the query point
    distances, indices = find_nearest_neighbors(nbrs, latent[query_index], n_neighbors)

    # Step 2: Compute inverse prevalence weights
    weights = 1.0 / (fg_counts + 1e-8)
    weights /= np.sum(weights)

    # Step 3: Compute neighbor weighted Tanimoto scores
    tanimoto_scores = np.zeros(len(indices), dtype=float)
    for i, idx in enumerate(indices):
        neighbor_fg = fg_labels[idx]
        tanimoto_scores[i] = weighted_tanimoto(fg_labels[query_index], neighbor_fg, weights)

    avg_weighted_tanimoto = np.mean(tanimoto_scores) if len(tanimoto_scores) > 0 else 0.0

    return avg_weighted_tanimoto


def extract_and_save_latents(
    model_path,
    dataloader,
    model_type,
    model_class,   # LightningModule class
    device="cpu",
    smiles_list=None,
    map_location=None,
    output_csv=None
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
        smiles_list (pd.Series or list[str], optional): Series or list of SMILES strings, one per datapoint.

    Returns:
        pd.DataFrame: Latents DataFrame including y if present and SMILES if provided.
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
            else:
                mu_z, logvar_z, mu_w, logvar_w = model.encode(x, y)
                z = model.reparameterize(mu_z, logvar_z)
                w = model.reparameterize(mu_w, logvar_w)

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

    # Add SMILES column if provided (pd.Series or list)
    if smiles_list is not None:
        smiles_arr = smiles_list.values if isinstance(smiles_list, pd.Series) else smiles_list
        if len(smiles_arr) != len(df):
            raise ValueError(f"Length of smiles_list ({len(smiles_arr)}) does not match number of rows ({len(df)})")
        df.insert(0, "smiles", smiles_arr)

    # Save CSV
    if output_csv is None:
        output_csv = f"latents/latents_{model_type}_{len(dataloader.dataset)}_{z_tensor.shape[1]}.csv"
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved latents (including y and smiles) to {output_csv}")

    return df

def evaluate_reconstructions(
    model_path,
    dataloader,
    model_type,
    model_class,  # LightningModule class
    device="cpu",
    map_location=None,
    thresholds=np.arange(0.05, 0.96, 0.05)  # Candidate thresholds for auto-selection
):
    """
    Evaluate reconstruction metrics for sparse binary vectors using a trained LightningModule (.ckpt).
    Automatically selects the threshold that maximizes F1 score.

    Supports BaseVAE, ConditionalVAE, ConditionalSubspaceVAE, DiscoverVAE.

    Returns:
        pd.DataFrame: DataFrame with reconstruction metrics for the dataset.
    """
    model_path = Path(model_path)
    print(f"Loading {model_type} model from {model_path}...")

    # Load model
    model = model_class.load_from_checkpoint(str(model_path), map_location=map_location).model
    model = model.to(device)
    model.eval()

    y_true_list, y_pred_probs_list = [], []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
                y = batch[1].to(device) if len(batch) > 1 and torch.is_tensor(batch[1]) else None
            else:
                x, y = batch.to(device)

            # Forward pass
            if model_type == "Base":
                x_logits, *_ = model(x)
                x_recon = torch.sigmoid(x_logits)
            elif model_type in ["CVAE", "CSVAE", "DISCoVeR"]:
                x_logits, *_ = model(x, y)
                x_recon = torch.sigmoid(x_logits)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Save ground truth and predicted probabilities
            y_true_list.append(x.cpu().numpy().astype(int))
            y_pred_probs_list.append(torch.sigmoid(x_recon).cpu().numpy())

    y_true_all = np.vstack(y_true_list)
    y_pred_probs_all = np.vstack(y_pred_probs_list)

    # -------------------------
    # Automatically select threshold
    # -------------------------
    best_threshold = None
    best_f1 = -1

    for t in thresholds:
        f1s = []
        for y_true_vec, y_prob_vec in zip(y_true_all, y_pred_probs_all):
            y_pred_vec = (y_prob_vec > t).astype(int)
            f1s.append(f1_score(y_true_vec, y_pred_vec, zero_division=0))
        avg_f1 = np.mean(f1s)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_threshold = t

    print(f"Selected threshold based on max F1: {best_threshold:.3f} (F1={best_f1:.4f})")

    # -------------------------
    # Compute final metrics using best threshold
    # -------------------------
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    for y_true_vec, y_prob_vec in zip(y_true_all, y_pred_probs_all):
        y_pred_vec = (y_prob_vec > best_threshold).astype(int)
        acc_list.append((y_true_vec == y_pred_vec).mean())
        prec_list.append(precision_score(y_true_vec, y_pred_vec, zero_division=0))
        rec_list.append(recall_score(y_true_vec, y_pred_vec, zero_division=0))
        f1_list.append(f1_score(y_true_vec, y_pred_vec, zero_division=0))

    metrics = {
        "threshold": best_threshold,
        "accuracy": np.mean(acc_list),
        "precision": np.mean(prec_list),
        "recall": np.mean(rec_list),
        "f1": np.mean(f1_list),
    }

    df = pd.DataFrame([metrics])

    # Save CSV
    output_csv = f"metrics/reconstruction_metrics_{str(model_path).split('/')[1]}.csv"
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved reconstruction metrics to {output_csv}")

    return df


def extract_prefixed_arrays(csv_path: str, prefixes: list[str]) -> pd.DataFrame:
    """
    Combines columns with specified prefixes into single array columns in a DataFrame.
    All other columns remain untouched, including a 'smiles' column if present.

    Parameters:
        csv_path (str): Path to the CSV file.
        prefixes (list[str]): List of prefixes to combine (e.g., ['z', 'w']).

    Returns:
        pd.DataFrame: DataFrame where each specified prefix is replaced with a column
                      containing NumPy arrays per row.
    """
    df = pd.read_csv(csv_path)

    combined_data = {}
    used_cols = set()

    # Combine specified prefixes
    for prefix in prefixes:
        cols = [col for col in df.columns if col.startswith(prefix)]
        if cols:
            combined_data[prefix] = df[cols].apply(lambda row: row.to_numpy(), axis=1)
            used_cols.update(cols)

    # Always keep 'smiles' column first if present
    cols_to_keep = []
    if 'smiles' in df.columns:
        cols_to_keep.append('smiles')
    # Keep all columns that were not combined and are not 'smiles'
    cols_to_keep += [col for col in df.columns if col not in used_cols and col != 'smiles']

    result_df = pd.concat([df[cols_to_keep], pd.DataFrame(combined_data)], axis=1)

    return result_df

def molecule_similarity(z_i, z_j, distance_metric="euclidean"):
    """
    Calculate similarity between two vectors based on the chosen distance metric.
    
    Parameters:
    z_i, z_j : np.ndarray
        Input vectors representing molecules.
    distance_metric : str
        "euclidean" or "cosine"
        
    Returns:
    float
        Similarity score between 0 and 1.
    """
    z_i = np.array(z_i)
    z_j = np.array(z_j)
    
    if distance_metric == "euclidean":
        d = np.linalg.norm(z_i - z_j, 2)
        similarity = 1 / (1 + d)
    elif distance_metric == "cosine":
        cos_sim = np.dot(z_i, z_j) / (np.linalg.norm(z_i) * np.linalg.norm(z_j))
        similarity = cos_sim
    else:
        raise ValueError("distance_metric must be 'euclidean' or 'cosine'")
    
    return similarity
