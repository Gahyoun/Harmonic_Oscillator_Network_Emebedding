import cupy as cp
import networkx as nx
from concurrent.futures import ProcessPoolExecutor

def HONE_GPU(adj_matrix, dim=2, num_steps=1000, learning_rate=0.01, seed=None):
    """
    Harmonic Oscillator Network Embedding (HONE) - GPU Accelerated Version
    
    This function embeds a network into a lower-dimensional space using a **spring-based gradient optimization** approach.
    The method is inspired by Hooke's Law, where edges act as springs pulling connected nodes toward their equilibrium distances.

    The optimization minimizes the total potential energy of the system using **gradient descent**.

    Parameters:
        adj_matrix (cp.ndarray): The adjacency matrix representing the network (must be a CuPy array).
        dim (int): The number of dimensions for the embedding space (default: 2).
        num_steps (int): The number of optimization iterations (default: 1000).
        learning_rate (float): The step size for gradient descent updates (default: 0.01).
        seed (int, optional): Random seed for reproducibility (default: None).

    Returns:
        cp.ndarray: Final node positions in the embedding space. Shape: (num_nodes, dim)
    """
    if seed is not None:
        cp.random.seed(seed)

    num_nodes = adj_matrix.shape[0]

    # Initialize random positions on GPU
    positions = cp.random.rand(num_nodes, dim)

    # Compute rest lengths (equilibrium distances) for edges (inverse of weights)
    rest_lengths = cp.zeros((num_nodes, num_nodes))
    mask = adj_matrix > 0
    rest_lengths[mask] = 1 / adj_matrix[mask]

    # Gradient optimization loop
    for _ in range(num_steps):
        gradients = cp.zeros((num_nodes, dim))

        # Compute gradient updates in parallel using GPU
        for node in range(num_nodes):
            mask = adj_matrix[node] > 0  # Consider only connected nodes
            delta = positions - positions[node]  # Displacement vectors
            distances = cp.linalg.norm(delta, axis=1) + 1e-8  # Prevent zero division
            r_0 = rest_lengths[node, mask]
            diff = (distances[mask] - r_0) / distances[mask]
            gradients[node] = cp.sum(diff[:, None] * delta[mask], axis=0)

        # Gradient descent update
        positions -= learning_rate * gradients

    return positions

def compute_distance_matrix_GPU(positions):
    """
    Compute the Euclidean distance matrix for final node positions using CuPy.

    Parameters:
        positions (cp.ndarray): Node positions from the embedding.

    Returns:
        cp.ndarray: Pairwise distance matrix of shape (num_nodes, num_nodes).
    """
    num_nodes = positions.shape[0]
    distance_matrix = cp.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        delta = positions - positions[i]
        distances = cp.linalg.norm(delta, axis=1)
        distance_matrix[i, :] = distances

    return distance_matrix

def parallel_HONE_GPU(adj_matrix, dim=2, num_steps=1000, learning_rate=0.01, seed_ensemble=10):
    """
    Perform multiple independent runs of HONE in parallel using multiprocessing and GPU acceleration.

    Parameters:
        adj_matrix (cp.ndarray): The adjacency matrix of the network (must be a CuPy array).
        dim (int): Number of dimensions for the embedding (default: 2).
        num_steps (int): Number of iterations for optimization (default: 1000).
        learning_rate (float): Learning rate for gradient descent (default: 0.01).
        seed_ensemble (int): Number of random initializations for ensemble computation (default: 10).

    Returns:
        list: List of CuPy arrays containing node positions for each ensemble run.
        cp.ndarray: 3D array of pairwise distance matrices (shape: (seed_ensemble, num_nodes, num_nodes)).
    """
    results = [None] * seed_ensemble

    # Convert adjacency matrix to CuPy
    adj_matrix = cp.asarray(adj_matrix)

    # Run HONE_GPU in parallel using multiple processes
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(HONE_GPU, adj_matrix, dim, num_steps, learning_rate, seed)
            for seed in range(seed_ensemble)
        ]
        for i, future in enumerate(futures):
            results[i] = future.result()

    # Extract node positions and compute distance matrices
    ensemble_positions = results
    distance_matrices = cp.array([compute_distance_matrix_GPU(result) for result in results])

    return ensemble_positions, distance_matrices

def HNI_GPU(distance_matrices):
    """
    Compute the Harmonic Network Inconsistency (HNI) value using CuPy.

    HNI quantifies the variance in pairwise node distances across multiple embeddings.
    A higher HNI indicates more inconsistency in node positions across different runs.

    Parameters:
        distance_matrices (cp.ndarray): 
            A 3D array of pairwise distance matrices from multiple embeddings (must be a CuPy array).
            Shape: (num_ensembles, num_nodes, num_nodes).

    Returns:
        float: The average variance of pairwise distances across all ensemble runs.
               Higher values indicate more inconsistency in the embeddings.
    """
    # Compute variance for each pair of nodes across different embeddings
    pairwise_variances = cp.var(distance_matrices, axis=0)  # Shape: (num_nodes, num_nodes)

    # Extract upper triangular part (excluding diagonal) to avoid redundancy
    upper_tri_indices = cp.triu_indices_from(pairwise_variances, k=1)
    upper_tri_variances = pairwise_variances[upper_tri_indices]

    # Compute mean variance (HNI), handling NaN cases
    hni_value = cp.nanmean(upper_tri_variances) if not cp.isnan(upper_tri_variances).all() else 0
    return float(hni_value)  # Convert to standard Python float for readability
