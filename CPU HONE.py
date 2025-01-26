import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor

def HONE_worker(adj_matrix, dim, iterations, tol, seed, dt, gamma):
    """
    Worker function for Harmonic Oscillator Network Embedding (HONE) using overdamped dynamics on the CPU.

    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix of the network.
        dim (int): Number of dimensions for the embedding space.
        iterations (int): Maximum number of iterations to run the embedding process.
        tol (float): Convergence tolerance for the total movement of positions.
        seed (int): Random seed for initialization.
        dt (float): Time step for the integration process.
        gamma (float): Damping coefficient for the overdamped dynamics.

    Returns:
        tuple:
            - positions (np.ndarray): Final positions of nodes in the embedding space (shape: num_nodes x dim).
            - distances (np.ndarray): Pairwise distances between nodes in the final embedding (shape: num_nodes x num_nodes).
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Initialize positions randomly in the embedding space and velocities to zero
    positions = np.random.rand(adj_matrix.shape[0], dim)
    velocities = np.zeros_like(positions)

    def calculate_forces(positions):
        """
        Calculate forces based on the harmonic oscillator model.

        Parameters:
            positions (np.ndarray): Current positions of nodes in the embedding space.

        Returns:
            np.ndarray: Forces acting on each node (shape: num_nodes x dim).
        """
        forces = np.zeros_like(positions)
        for i in range(len(positions)):
            # Calculate displacement vectors from node i to all other nodes
            delta = positions - positions[i]
            # Compute distances between node i and others
            distances = np.linalg.norm(delta, axis=1)
            # Mask to avoid division by zero (self-loops)
            mask = distances > 1e-6
            # Compute forces based on the adjacency matrix and normalized displacements
            forces[i] = np.sum(
                adj_matrix[i, mask][:, None] * (delta[mask] / distances[mask, None]),
                axis=0
            )
        return forces

    # Iterative integration process
    for _ in range(iterations):
        # Calculate forces for the current positions
        forces = calculate_forces(positions)
        # Update velocities based on overdamped dynamics
        velocities = -forces / gamma
        # Update positions using the calculated velocities
        new_positions = positions + velocities * dt
        # Calculate total movement to check convergence
        total_movement = np.sum(np.linalg.norm(new_positions - positions, axis=1))
        if total_movement < tol:  # Convergence condition
            break
        # Update positions for the next iteration
        positions = new_positions

    # Calculate the pairwise distances in the final embedding
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    return positions, distances

def HONE(G, dim=2, iterations=100, seed_ensemble=100, tol=1e-4, dt=0.01, gamma=1.0):
    """
    Perform Harmonic Oscillator Network Embedding (HONE) for a given graph.

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
    # Convert the graph to an adjacency matrix (weighted or unweighted)
    if nx.is_weighted(G):
        adj_matrix = np.asarray(nx.to_numpy_array(G, weight="weight"))
    else:
        adj_matrix = np.asarray(nx.to_numpy_array(G))
        adj_matrix[adj_matrix > 0] = 1  # Convert to unweighted if needed

    results = [None] * seed_ensemble

    # Use multithreading for parallel processing of ensemble calculations
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(HONE_worker, adj_matrix, dim, iterations, tol, seed, dt, gamma)
            for seed in range(seed_ensemble)
        ]
        for i, future in enumerate(futures):
            results[i] = future.result()

    # Extract node positions and distance matrices from the results
    ensemble_positions = [result[0] for result in results]
    distance_matrices = np.array([result[1] for result in results])

    return ensemble_positions, distance_matrices

def HNI(distance_matrices):
    """
    Calculate the Harmonic Network Inconsistency (HNI) to quantify variance across ensembles.

    Parameters:
        distance_matrices (np.ndarray): Array of pairwise distance matrices for each ensemble (shape: seed_ensemble x num_nodes x num_nodes).

    Returns:
        float: Average variance of pairwise distances across ensembles.
    """
    # Compute the variance of distances for each pair of nodes across ensembles
    pairwise_variances = np.var(distance_matrices, axis=0)  # Shape: (num_nodes x num_nodes)
    # Extract the upper triangular part of the variance matrix (excluding the diagonal)
    upper_tri_indices = np.triu_indices_from(pairwise_variances, k=1)
    upper_tri_variances = pairwise_variances[upper_tri_indices]
    # Calculate and return the average variance
    return np.mean(upper_tri_variances)
