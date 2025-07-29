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
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import math

### ----------------------Functions for scaffold and EFG computation---------------------- ###

PATT = Chem.MolFromSmarts("[$([D1]=[*])]")
REPL = Chem.MolFromSmiles("*")

def get_scaffold(mol, real_bm=True, use_csk=False, use_bajorath=False):
    """
    Extracts the Murcko scaffold from a molecule, with options for Bajorath modification and CSK canonization.
    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        real_bm (bool): Whether to apply real Bajorath modification.
        use_csk (bool): Whether to canonize the scaffold using CSK.
        use_bajorath (bool): Whether to apply Bajorath modification.
    Returns:
        rdkit.Chem.Mol: Scaffold molecule.
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

def save_chunk_results(dir, chunk_idx, results):
    """
    Saves results of a chunked computation to a pickle file.
    Args:
        dir (str): Directory to save the file.
        chunk_idx (int): Chunk index.
        results (object): Results to save.
    """
    with open(os.path.join(dir, f"chunk_{chunk_idx}.pkl"), "wb") as f:
        pickle.dump(results, f)

def safe_mol_from_smiles(smiles):
    """
    Safely converts a SMILES string to an RDKit Mol object.
    Args:
        smiles (str): SMILES string.
    Returns:
        rdkit.Chem.Mol or None: Molecule object or None if conversion fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None

def compute_efgs(mol):
    """
    Computes extended functional groups (EFGs) for a molecule.
    Args:
        mol (rdkit.Chem.Mol): Input molecule.
    Returns:
        list: List of EFG SMILES strings.
    """
    _, _, psmis, _ = efgs.get_dec_fgs(mol)
    return psmis

def compute_efgs_safe(mol):
    """
    Safely computes EFGs for a molecule, handling exceptions.
    Args:
        mol (rdkit.Chem.Mol): Input molecule.
    Returns:
        list or None: List of EFG SMILES or None if computation fails.
    """
    try:
        return compute_efgs(mol)
    except Exception as e:
        print(f"Error computing EFGs for molecule: {mol}. Error: {e}")
        return None

def compute_scaffold_safe(mol):
    """
    Safely computes the scaffold for a molecule, handling exceptions.
    Args:
        mol (rdkit.Chem.Mol): Input molecule.
    Returns:
        rdkit.Chem.Mol or None: Scaffold molecule or None if computation fails.
    """
    try:
        return get_scaffold(mol)
    except Exception:
        return None

### ----------------------Functions for fingerprint generation and functional group encoding---------------------- ###

def create_mfpgen():
    """
    Creates a Morgan fingerprint generator with radius 2 and 2048 bits.
    Returns:
        rdkit.Chem.rdFingerprintGenerator.MorganFingerprintGenerator: Fingerprint generator.
    """
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

mfpgen = create_mfpgen()

def mol_to_fingerprint(mol):
    """
    Converts a molecule to a Morgan fingerprint.
    Args:
        mol (rdkit.Chem.Mol): Input molecule.
    Returns:
        rdkit.DataStructs.cDataStructs.ExplicitBitVect or None: Fingerprint or None if input is None.
    """
    if mol is None:
        return None
    return mfpgen.GetFingerprint(mol)

def smiles_to_mol(smiles):
    """
    Converts a SMILES string to an RDKit Mol object, handling exceptions.
    Args:
        smiles (str): SMILES string.
    Returns:
        rdkit.Chem.Mol or None: Molecule object or None if conversion fails.
    """
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception as e:
        print(f"Error converting SMILES {smiles}: {e}")
        return None

def fg_to_array(fgs, fg_list):
    """
    Encodes the presence of functional groups as a binary array.
    Args:
        fgs (list): List of functional groups present.
        fg_list (list): List of all possible functional groups.
    Returns:
        np.ndarray: Binary array indicating presence of each functional group.
    """
    fg_array = np.zeros(len(fg_list), dtype=int)
    if isinstance(fgs, list):
        for fg in fgs:
            if fg in fg_list:
                fg_array[fg_list.index(fg)] = 1
    return fg_array

def fp_to_array(fp):
    """
    Converts an RDKit fingerprint to a numpy array.
    Args:
        fp (rdkit.DataStructs.cDataStructs.ExplicitBitVect): Fingerprint.
    Returns:
        np.ndarray: Numpy array representation of the fingerprint.
    """
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

### ----------------------Functions and Classes for VAE Creation and Testing---------------------- ###


def vae_loss(recon_x, x, mu, log_var, beta=1):
    """
    Computes the loss for a Variational Autoencoder (VAE).
    Args:
        recon_x (torch.Tensor): Reconstructed input.
        x (torch.Tensor): Original input.
        mu (torch.Tensor): Mean of latent distribution.
        log_var (torch.Tensor): Log variance of latent distribution.
        beta (float): Weight for KL divergence term.
    Returns:
        tuple: (total_loss, bce_loss, kld_loss) per batch.
    """
    bce = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (bce + beta * kld) / x.size(0), bce / x.size(0), kld / x.size(0)

def visualize_latent_space(model, dataset, fg_indicies=[0, 1, 2, 3], method='tsne', sample_size=None):
    """
    Visualizes the latent space of a VAE model using dimensionality reduction and colors by functional group combinations.
    Args:
        model (torch.nn.Module): Trained VAE model.
        dataset (torch.utils.data.Dataset): Dataset for visualization.
        fg_indicies (list): Indices of functional groups to use for coloring.
        method (str): Dimensionality reduction method ('tsne', 'pca', 'umap').
        sample_size (int or None): Number of samples to plot (randomly selected).
    """
    model.eval()
    latents = []
    fg_labels = []

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            x, fg_vecs, _ = batch
            _, mu, _ = model(x)
            latents.append(mu.cpu())
            fg_labels.append(fg_vecs.cpu())

    latents = torch.cat(latents).numpy()
    fg_labels = torch.cat(fg_labels).numpy()

    if sample_size and sample_size < len(latents):
        indices = np.random.choice(len(latents), sample_size, replace=False)
        latents = latents[indices]
        fg_labels = fg_labels[indices]

    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'umap':
        reducer = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid method. Choose from 'tsne', 'pca', 'umap'.")

    latents_2d = reducer.fit_transform(latents)

    combined_labels = []
    for i in range(len(fg_labels)):
        label_parts = []
        for fg_idx in fg_indicies:
            label_parts.append(f"{int(fg_labels[i, fg_idx])}")
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

def perform_umap_on_fingerprints(data, fingerprint_col='fingerprint_array', fg_col='fg_array', 
                                 fg_indicies=[0, 1, 2, 3], n_neighbors=15, min_dist=0.1, metric='euclidean'):
    """
    Performs UMAP dimensionality reduction on molecular fingerprints and visualizes them colored by functional group combinations.
    Args:
        data (pd.DataFrame): DataFrame containing fingerprint and functional group arrays.
        fingerprint_col (str): Column name for fingerprint arrays.
        fg_col (str): Column name for functional group arrays.
        fg_indicies (list): Indices of functional groups to use for coloring.
        n_neighbors (int): UMAP parameter for local neighborhood size.
        min_dist (float): UMAP parameter for minimum distance between points.
        metric (str): Distance metric for UMAP.
    Returns:
        np.ndarray: UMAP embeddings of the fingerprints.
    """
    fingerprints = np.array(data[fingerprint_col].tolist())
    fg_labels = np.array(data[fg_col].tolist())

    umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    umap_embeddings = umap_model.fit_transform(fingerprints)

    combined_labels = []
    for i in range(len(fg_labels)):
        label_parts = []
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

def f1_score(y_true, y_pred):
    """
    Calculates the F1 score, precision, and recall for binary classification.
    Args:
        y_true (np.ndarray): Ground truth binary array.
        y_pred (np.ndarray): Predicted binary array.
    Returns:
        tuple: (f1, precision, recall)
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

def calculate_reconstruction_quality(model, dataset, threshold=0.5):
    """
    Evaluates the reconstruction quality of a model on a given dataset using Tanimoto similarity, F1 score, precision, recall, and uniqueness.
    Args:
        model (torch.nn.Module): Model to evaluate, expected to output reconstructed fingerprints.
        dataset (torch.utils.data.Dataset): Dataset containing input fingerprints and labels.
        threshold (float): Threshold for binarizing reconstructed fingerprints.
    Returns:
        tuple: (tanimoto_scores, f1_scores, precision_scores, recall_scores, uniqueness)
    """
    model.eval()
    all_fingerprints = []
    all_reconstructions = []
    all_fg_labels = []

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            x, fg_vecs, _ = batch
            recon_x, _, _ = model(x)
            all_fingerprints.append(x.cpu())
            all_reconstructions.append(recon_x.cpu())
            all_fg_labels.append(fg_vecs.cpu())

    all_fingerprints = torch.cat(all_fingerprints).numpy()
    all_reconstructions = torch.cat(all_reconstructions).numpy()
    all_fg_labels = torch.cat(all_fg_labels).numpy()

    # Calculate Tanimoto similarity
    tanimoto_scores = []
    for i in range(len(all_fingerprints)):
        fp1 = Chem.DataStructs.cDataStructs.CreateFromBitString(''.join(map(str, all_fingerprints[i].astype(int))))
        fp2 = Chem.DataStructs.cDataStructs.CreateFromBitString(''.join(map(str, (all_reconstructions[i] > threshold).astype(int))))
        tanimoto_scores.append(DataStructs.TanimotoSimilarity(fp1, fp2))

    tanimoto_scores = np.array(tanimoto_scores)

    # Calculate F1 score, precision, and recall for each fingerprint
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for i in range(len(all_fingerprints)):
        y_true = all_fingerprints[i]
        y_pred = all_reconstructions[i] > threshold
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        f1, precision, recall = f1_score(y_true, y_pred)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    f1_scores = np.array(f1_scores)
    precision_scores = np.array(precision_scores)
    recall_scores = np.array(recall_scores)

    # Calculate uniqueness of thresholded reconstructions
    unique_reconstructions = np.unique(all_reconstructions[all_reconstructions > threshold], axis=0)
    uniqueness = len(unique_reconstructions) / len(all_reconstructions[all_reconstructions > threshold])

    return tanimoto_scores, f1_scores, precision_scores, recall_scores, uniqueness

# get average latent vector for each functional group
def average_latent_vector(model, dataset, fg_col='fg_array', fg_indicies=[0, 1, 2, 3]):
    model.eval()
    fg_latents = {fg_idx: [] for fg_idx in fg_indicies}

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            x, fg_vecs, _ = batch
            _, mu, _ = model(x)
            mu = mu.cpu().numpy()
            fg_vecs = fg_vecs.cpu().numpy()

            for i in range(len(mu)):
                for fg_idx in fg_indicies:
                    if fg_vecs[i, fg_idx] == 1:  # Check if the functional group is present
                        fg_latents[fg_idx].append(mu[i])

    # Calculate average latent vector for each functional group
    avg_latents = {fg_idx: np.mean(fg_latents[fg_idx], axis=0) if fg_latents[fg_idx] else None for fg_idx in fg_indicies}
    
    return avg_latents

def plot_average_latent_vectors(avg_latents):
    # Convert dict to sorted list for consistent plotting
    fg_indices = sorted(avg_latents.keys())
    latent_dim = len(next(iter(avg_latents.values())))  # infer latent dim

    # Create matrix: rows = FG, cols = latent dimensions
    data = np.array([avg_latents[fg] if avg_latents[fg] is not None else np.zeros(latent_dim)
                     for fg in fg_indices])
    
    # Plot grouped bars
    x = np.arange(latent_dim)  # latent dimension indices
    bar_width = 0.8 / len(fg_indices)  # width for grouped bars

    plt.figure(figsize=(12, 6))
    for i, fg in enumerate(fg_indices):
        plt.bar(x + i * bar_width, data[i], width=bar_width, label=f'FG{fg}')

    plt.title('Average Latent Vectors by Functional Group')
    plt.xlabel('Latent Dimension Index')
    plt.ylabel('Average Latent Value')
    plt.xticks(x + bar_width * (len(fg_indices)-1) / 2, range(latent_dim))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_distributions(plot_info_list, bins=50, figsize_per_plot=(5, 4)):
    """
    Generates histogram plots for any number of datasets, with customizable titles and x-axis labels.

    Args:
        plot_info_list (list of dict): A list where each dictionary represents one plot and must contain:
                                        - 'data' (list or array-like): The numerical data for the histogram.
                                        - 'title' (str): The title for this specific plot.
                                        - 'xlabel' (str): The x-axis label for this specific plot.
                                        Example: [{'data': [...], 'title': 'Tanimoto', 'xlabel': 'Similarity'}]
        bins (int, optional): Number of bins for all histograms. Defaults to 50.
        figsize_per_plot (tuple, optional): Base figure size (width, height) in inches for each subplot.
                                            The total figure size will be calculated based on this. Defaults to (5, 4).
    """
    num_plots = len(plot_info_list)

    if num_plots == 0:
        print("No plot information provided.")
        return

    # Determine grid layout for subplots
    # We'll try to keep a maximum of 3 columns for readability
    ncols = min(num_plots, 3)
    nrows = math.ceil(num_plots / ncols)

    # Calculate total figure size based on the number of rows and columns
    total_figsize_width = ncols * figsize_per_plot[0]
    total_figsize_height = nrows * figsize_per_plot[1]

    # Create the figure and a grid of subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(total_figsize_width, total_figsize_height))

    # If there's only one subplot, axes is not an array, so make it iterable
    if num_plots == 1:
        axes = [axes]
    # If it's a 1D array (e.g., 1 row, multiple columns or vice versa), flatten it
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    # Otherwise, axes is already a 2D array and can be flattened for easy iteration
    else:
        axes = axes.flatten()

    # Iterate through the provided plot information and create each subplot
    for i, plot_info in enumerate(plot_info_list):
        ax = axes[i] # Get the current subplot axis

        data = plot_info['data']
        title = plot_info['title']
        xlabel = plot_info['xlabel']

        sns.histplot(data, bins=bins, kde=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')

    # Hide any unused subplots if the grid is larger than num_plots
    for j in range(num_plots, nrows * ncols):
        fig.delaxes(axes[j]) # Remove unused axes

    plt.tight_layout() # Adjust subplot parameters for a tight layout
    plt.show()

### ----------------------Functions for latent space metric---------------------- ###
# Get nearest neighbors in latent space
def get_nearest_neighbors(model, dataset, latents=None, n_neighbors=50):
    model.eval()
    if latents is None:
        latents = []
        with torch.no_grad():
            for x, _, _ in DataLoader(dataset, batch_size=64, shuffle=False):
                _, mu, _ = model(x)
                latents.append(mu.cpu())
        latents = torch.cat(latents).numpy()

    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(latents)
    return nbrs, latents

def find_nearest_neighbors(nbrs, query_latent, n_neighbors=50):
    distances, indices = nbrs.kneighbors(query_latent.reshape(1, -1), n_neighbors=n_neighbors)
    return distances.flatten(), indices.flatten()

def get_fg_counts(dataset, fg_col='fg_array'):
    """Get counts of functional groups in the dataset."""
    df = dataset.data
    fg_list = df[fg_col].tolist()
    fg_counts = {i: 0 for i in range(len(fg_list[0]))}  # Initialize counts for each functional group
    for fg in fg_list:
        for i, val in enumerate(fg):
            if val == 1:
                fg_counts[i] += 1
    fg_counts = np.array(list(fg_counts.values()))
    fg_counts = fg_counts.astype(float)  # Ensure counts are float for normalization
    return fg_counts / len(fg_list)  # Normalize by total number of samples

# Calculate percentage of nearest neighbors with the same functional group
def metric(nbrs, dataset, query_index, latent, fg_counts, fg_col='fg_array', fg_indicies=[0, 1, 2, 3], n_neighbors=50):
    query_fg = dataset.data.iloc[query_index][fg_col]
    distances, indices = find_nearest_neighbors(nbrs, latent[query_index], n_neighbors)

    same_fg_count = 0
    for idx in indices:
        neighbor_fg = dataset.data.iloc[idx][fg_col]
        if any(neighbor_fg[fg_idx] == query_fg[fg_idx] == 1 for fg_idx in fg_indicies):
            same_fg_count += 1
    
    # Normalize by the prevalence of the functional groups in the dataset
    fg_prevalence = np.mean([fg_counts[fg_idx] for fg_idx in fg_indicies if query_fg[fg_idx] == 1])
    if fg_prevalence == 0:
        return 0.0

    same_fg_percentage = (same_fg_count / n_neighbors) / fg_prevalence
    return same_fg_percentage

