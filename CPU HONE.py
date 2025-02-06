from concurrent.futures import ThreadPoolExecutor
import numpy as np
import networkx as nx

def HONE_worker(adj_matrix, dim, iterations, tol, seed, dt, gamma):
    """
    Worker function for Harmonic Oscillator Network Embedding (HONE) using overdamped dynamics on CPU.

    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the network.
        dim (int): Number of dimensions for the embedding space.
        iterations (int): Maximum number of iterations to run the embedding process.
        tol (float): Convergence tolerance for the total movement of positions.
        seed (int): Random seed for initializing positions.
        dt (float): Time step for the integration process.
        gamma (float): Damping coefficient for the overdamped dynamics.

    Returns:
        tuple:
            - positions (np.ndarray): Final positions of nodes in the embedding space (shape: num_nodes x dim).
            - distances (np.ndarray): Pairwise distances between nodes in the final embedding (shape: num_nodes x num_nodes).
    """
    np.random.seed(seed)

    # Initialize positions randomly in the embedding space and velocities to zero
    num_nodes = adj_matrix.shape[0]
    positions = np.random.rand(num_nodes, dim)
    velocities = np.zeros_like(positions)

    def calculate_forces(positions):
        """
        Calculate forces based on the harmonic oscillator model using CPU computations.

        Parameters:
            positions (np.ndarray): Current positions of nodes in the embedding space.

        Returns:
            np.ndarray: Forces acting on each node (shape: num_nodes x dim).
        """
        forces = np.zeros_like(positions)
        for i in range(num_nodes):
            delta = positions - positions[i]  # Displacement vectors
            distances = np.linalg.norm(delta, axis=1)  # Compute distances
            mask = distances != 0  # Exclude self-distance

            # Compute forces only for connected nodes
            normalized_displacement = delta[mask] / distances[mask, None]
            forces[i] = np.sum(adj_matrix[i, mask][:, None] * normalized_displacement, axis=0)
        return forces

    # Iterative optimization process
    for _ in range(iterations):
        forces = calculate_forces(positions)
        velocities = -forces / gamma  # Overdamped dynamics
        new_positions = positions + velocities * dt  # Update positions

        # Convergence check
        total_movement = np.sum(np.linalg.norm(new_positions - positions, axis=1))
        if total_movement < tol:
            break
        positions = new_positions  # Update positions

    # Compute final pairwise distances
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)

    return positions, distances

def HONE(G, dim=2, iterations=100, seed_ensemble=100, tol=1e-4, dt=0.01, gamma=1.0):
    """
    Perform Harmonic Oscillator Network Embedding (HONE) using CPU-based overdamped dynamics.

    Parameters:
        G (networkx.Graph): Input graph to be embedded.
        dim (int): Number of dimensions for the embedding space.
        iterations (int): Maximum number of iterations for the embedding process.
        seed_ensemble (int): Number of random initializations (seeds) for ensemble calculation.
        tol (float): Convergence tolerance for the total movement of positions.
        dt (float): Time step for the integration process.
        gamma (float): Damping coefficient for the overdamped dynamics.

    Returns:
        tuple:
            - ensemble_positions (list of np.ndarray): List of node positions for each ensemble (length: seed_ensemble).
            - distance_matrices (np.ndarray): Array of pairwise distance matrices for each ensemble (shape: seed_ensemble x num_nodes x num_nodes).
    """
    # Convert graph to an adjacency matrix
    adj_matrix = nx.to_numpy_array(G, weight="weight")
    if not nx.is_weighted(G):
        adj_matrix[adj_matrix > 0] = 1  # Convert to binary adjacency for unweighted graphs

    results = [None] * seed_ensemble  # Preallocate result storage

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(HONE_worker, adj_matrix, dim, iterations, tol, seed, dt, gamma)
            for seed in range(seed_ensemble)
        ]
        for i, future in enumerate(futures):
            results[i] = future.result()

    # Extract results
    ensemble_positions = [result[0] for result in results]
    distance_matrices = np.array([result[1] for result in results])

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
    
