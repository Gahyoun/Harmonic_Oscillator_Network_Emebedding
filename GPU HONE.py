import numpy as np
import cupy as cp
import networkx as nx

def HONE_worker(adj_matrix, dim, iterations, tol, seed, dt, gamma):
    """
    GPU-accelerated version of Harmonic Oscillator Network Embedding (HONE) using CuPy.
    Numerical integration follows the equation:
        r_i^(t+1) = r_i^(t) - dt * sum(w_ij * (r_i^(t) - r_j^(t))), where w_ij is the weight of the edge.

    Parameters:
        adj_matrix (cp.ndarray): Adjacency matrix of the network (weights as spring constants, w_ij).
        dim (int): Dimensionality of the embedding space.
        iterations (int): Maximum number of iterations for the embedding process.
        tol (float): Convergence threshold for the total movement of positions.
        seed (int): Random seed for reproducibility.
        dt (float): Time step for numerical integration.
        gamma (float): Damping coefficient (not explicitly used here due to direct position updates).

    Returns:
        tuple:
            - positions (cp.ndarray): Final positions of nodes in the embedding space (shape: num_nodes x dim).
            - distances (cp.ndarray): Pairwise distances between nodes in the final embedding (shape: num_nodes x num_nodes).
    """
    # Set the random seed for reproducibility
    cp.random.seed(seed)

    num_nodes = adj_matrix.shape[0]

    # Initialize positions randomly within the embedding space
    positions = cp.random.rand(num_nodes, dim) * 0.1

    def calculate_forces(positions):
        """
        GPU-accelerated force calculation based on adjacency matrix.

        Parameters:
            positions (cp.ndarray): Current positions of nodes (num_nodes x dim).

        Returns:
            cp.ndarray: Forces acting on each node (num_nodes x dim).
        """
        # Compute weighted displacements for all nodes simultaneously
        forces = -cp.matmul(adj_matrix, positions) + cp.sum(adj_matrix, axis=1, keepdims=True) * positions
        return forces

    # Iterative numerical integration process
    for _ in range(iterations):
        # Compute forces for the current positions
        forces = calculate_forces(positions)

        # Update positions based on the given equation
        new_positions = positions - dt * forces

        # Check convergence: If total movement is below the tolerance, stop early
        total_movement = cp.sum(cp.linalg.norm(new_positions - positions, axis=1))
        if total_movement < tol:
            break

        # Update positions for the next iteration
        positions = new_positions

    # Compute the pairwise distances in the final embedding
    distances = cp.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    return positions, distances

def HONE(G, dim=2, iterations=100, seed_ensemble=100, tol=1e-4, dt=0.01, gamma=1.0):
    """
    Perform Harmonic Oscillator Network Embedding (HONE) using GPU-based overdamped dynamics.

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
    # Convert the graph to an adjacency matrix and move it to the GPU
    adj_matrix = cp.asarray(nx.to_numpy_array(G, weight="weight"))
    if not nx.is_weighted(G):
        adj_matrix[adj_matrix > 0] = 1  # Convert to binary adjacency for unweighted graphs

    # Create a list of CUDA streams for asynchronous execution
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(seed_ensemble)]
    results = [None] * seed_ensemble  # Preallocate results array

    # Launch HONE_worker for each seed using separate CUDA streams
    for seed, stream in zip(range(seed_ensemble), streams):
        results[seed] = HONE_worker(adj_matrix, dim, iterations, tol, seed, dt, gamma, stream)

    # Wait for all CUDA streams to finish execution
    cp.cuda.Stream.null.synchronize()

    # Extract results from the GPU
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
