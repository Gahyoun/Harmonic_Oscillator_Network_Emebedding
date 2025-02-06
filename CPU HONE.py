import numpy as np
import networkx as nx
from concurrent.futures import ProcessPoolExecutor

def HONE(adj_matrix, dim=2, num_steps=1000, learning_rate=0.01, seed=None):
    """
    Harmonic Oscillator Network Embedding (HONE)

    This function embeds a network into a lower-dimensional space using a **spring-based gradient optimization** approach.
    The method is inspired by Hooke's Law, where edges act as springs pulling connected nodes toward their equilibrium distances.

    The optimization minimizes the total potential energy of the system using **gradient descent**.

    Parameters:
        adj_matrix (np.ndarray): The adjacency matrix representing the network.
        dim (int): The number of dimensions for the embedding space (default: 2).
        num_steps (int): The number of optimization iterations (default: 1000).
        learning_rate (float): The step size for gradient descent updates (default: 0.01).
        seed (int, optional): Random seed for reproducibility (default: None).

    Returns:
        dict: Final node positions in the embedding space.
              Keys are node indices, values are NumPy arrays of shape (dim,).
    """
    if seed is not None:
        np.random.seed(seed)

    num_nodes = adj_matrix.shape[0]

    # Initialize random positions
    positions = {node: np.random.rand(dim) for node in range(num_nodes)}

    # Compute rest lengths (equilibrium distances) for edges (inverse of weights)
    rest_lengths = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] > 0:  # Only consider existing edges
                rest_lengths[(i, j)] = 1 / adj_matrix[i, j]

    # Gradient optimization loop
    for _ in range(num_steps):
        new_positions = {}
        for node in range(num_nodes):
            gradient = np.zeros(dim)
            for neighbor in range(num_nodes):
                if adj_matrix[node, neighbor] > 0:
                    distance = np.linalg.norm(positions[node] - positions[neighbor]) + 1e-8  # Prevent zero division
                    r_0 = rest_lengths[(node, neighbor)]
                    diff = (distance - r_0) / distance  # Gradient of spring potential
                    gradient += diff * (positions[node] - positions[neighbor])

            # Gradient descent update
            new_positions[node] = positions[node] - learning_rate * gradient

        positions = new_positions  # Apply updates

    return positions

def compute_distance_matrix(positions):
    """
    Compute the Euclidean distance matrix for final node positions.

    Parameters:
        positions (dict): Node positions from the embedding.

    Returns:
        np.ndarray: Pairwise distance matrix of shape (num_nodes, num_nodes).
    """
    nodes = list(positions.keys())
    num_nodes = len(nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = np.linalg.norm(positions[nodes[i]] - positions[nodes[j]])
            distance_matrix[i, j] = distance_matrix[j, i] = distance

    return distance_matrix

def parallel_HONE(adj_matrix, dim=2, num_steps=1000, learning_rate=0.01, seed_ensemble=10):
    """
    Perform multiple independent runs of HONE in parallel using multiprocessing.

    Parameters:
        adj_matrix (np.ndarray): The adjacency matrix of the network.
        dim (int): Number of dimensions for the embedding (default: 2).
        num_steps (int): Number of iterations for optimization (default: 1000).
        learning_rate (float): Learning rate for gradient descent (default: 0.01).
        seed_ensemble (int): Number of random initializations for ensemble computation (default: 10).

    Returns:
        list: List of node position dictionaries for each ensemble run.
        np.ndarray: 3D array of pairwise distance matrices (shape: (seed_ensemble, num_nodes, num_nodes)).
    """
    results = [None] * seed_ensemble

    # Run HONE in parallel using multiple processes
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(HONE, adj_matrix, dim, num_steps, learning_rate, seed)
            for seed in range(seed_ensemble)
        ]
        for i, future in enumerate(futures):
            results[i] = future.result()

    # Extract node positions and compute distance matrices
    ensemble_positions = results
    distance_matrices = np.array([compute_distance_matrix(result) for result in results])

    return ensemble_positions, distance_matrices

def HNI(distance_matrices):
    """
    Compute the Harmonic Network Inconsistency (HNI) value.

    HNI quantifies the variance in pairwise node distances across multiple embeddings.
    A higher HNI indicates more inconsistency in node positions across different runs.

    Parameters:
        distance_matrices (np.ndarray): 
            A 3D array of pairwise distance matrices from multiple embeddings.
            Shape: (num_ensembles, num_nodes, num_nodes).

    Returns:
        float: The average variance of pairwise distances across all ensemble runs.
               Higher values indicate more inconsistency in the embeddings.
    """
    # Compute variance for each pair of nodes across different embeddings
    pairwise_variances = np.var(distance_matrices, axis=0)  # Shape: (num_nodes, num_nodes)

    # Extract upper triangular part (excluding diagonal) to avoid redundancy
    upper_tri_indices = np.triu_indices_from(pairwise_variances, k=1)
    upper_tri_variances = pairwise_variances[upper_tri_indices]

    # Compute mean variance (HNI), handling NaN cases
    hni_value = np.nanmean(upper_tri_variances) if not np.isnan(upper_tri_variances).all() else 0
    return hni_value
    
