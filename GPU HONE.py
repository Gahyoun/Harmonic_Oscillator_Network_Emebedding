import cupy as cp
import networkx as nx
import numpy as np

def HONE_worker_gpu(adj_matrix, dim, iterations, tol, seed, dt, gamma):
    """
    GPU-accelerated worker function for Harmonic Oscillator Network Embedding (HONE) 
    using CuPy for parallel computation.

    Parameters:
        adj_matrix (cp.ndarray): Adjacency matrix (weights as spring constants).
        dim (int): Number of embedding dimensions.
        iterations (int): Maximum number of iterations.
        tol (float): Convergence threshold.
        seed (int): Random seed.
        dt (float): Time step.
        gamma (float): Damping coefficient for the overdamped dynamics.

    Returns:
        tuple:
            - positions (cp.ndarray): Final positions of nodes in the embedding space (num_nodes x dim).
            - distances (cp.ndarray): Pairwise node distances in the final embedding (num_nodes x num_nodes).
    """
    cp.random.seed(seed)  # Set random seed for reproducibility
    num_nodes = adj_matrix.shape[0]  # Number of nodes in the network

    # Normalize adjacency matrix weights to avoid numerical instability
    max_weight = cp.max(adj_matrix)
    if max_weight > 0:
        adj_matrix = adj_matrix / max_weight

    # Initialize node positions with small perturbations (on GPU)
    positions = cp.random.rand(num_nodes, dim) * 0.1
    optimal_distances = cp.copy(adj_matrix)  # Initialize optimal distances using adjacency weights

    def compute_optimal_distances(positions):
        """ Compute optimal distances between nodes based on network structure. """
        nonlocal optimal_distances
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:
                    k_ij = adj_matrix[i, j]  # Spring constant (edge weight)
                    r_ij = cp.linalg.norm(positions[j] - positions[i])  # Current node distance
                    gradient = -k_ij * (r_ij - optimal_distances[i, j])  # Gradient of energy function
                    optimal_distances[i, j] = r_ij - gradient / k_ij  # Update optimal distance
        return optimal_distances

    def compute_forces(positions, optimal_distances):
        """ Compute restoring forces acting on each node using GPU parallelization. """
        forces = cp.zeros_like(positions)  # Initialize forces array on GPU
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:
                    k_ij = adj_matrix[i, j]  # Spring constant
                    r_ij = positions[j] - positions[i]  # Vector displacement
                    distance = cp.linalg.norm(r_ij)  # Scalar distance

                    # Avoid division by zero when calculating unit vector
                    unit_vector = r_ij / distance if distance > 1e-6 else cp.zeros_like(r_ij)

                    # Compute force magnitude based on Hooke's Law
                    force_magnitude = -k_ij * (distance - optimal_distances[i, j])
                    forces[i] += force_magnitude * unit_vector  # Apply force to node i
        return forces

    # Iterative simulation loop for convergence
    for _ in range(iterations):
        optimal_distances = compute_optimal_distances(positions)
        forces = compute_forces(positions, optimal_distances)

        # Update node positions using overdamped dynamics
        new_positions = positions - (forces / gamma) * dt  

        # Check convergence based on total movement
        total_movement = cp.sum(cp.linalg.norm(new_positions - positions, axis=1))
        if total_movement < tol:
            break

        positions = new_positions  # Update positions for next iteration

    # Compute final pairwise distances between nodes
    distances = cp.linalg.norm(positions[:, None] - positions[None, :], axis=2)

    return positions, distances

def HONE_gpu(G, dim=2, iterations=100, seed_ensemble=100, tol=1e-4, dt=0.01, gamma=1):
    """
    Perform GPU-accelerated Harmonic Oscillator Network Embedding (HONE) for a given graph.

    Parameters:
        G (networkx.Graph): Input network graph.
        dim (int): Number of dimensions for the embedding space.
        iterations (int): Maximum number of iterations for the embedding process.
        seed_ensemble (int): Number of random initializations (seeds) for ensemble calculation.
        tol (float): Convergence tolerance based on total movement.
        dt (float): Time step for numerical integration.
        gamma (float): Damping coefficient for overdamped dynamics.

    Returns:
        tuple:
            - ensemble_positions (list of cp.ndarray): Node positions for each ensemble (length: seed_ensemble).
            - distance_matrices (cp.ndarray): Pairwise distance matrices for each ensemble (shape: seed_ensemble x num_nodes x num_nodes).
    """
    # Convert NetworkX graph to CuPy adjacency matrix
    if nx.is_weighted(G):
        adj_matrix = cp.asarray(nx.to_numpy_array(G, weight="weight"))
        max_weight = cp.max(adj_matrix)
        if max_weight > 0:
            adj_matrix = adj_matrix / max_weight  # Normalize weights
    else:
        adj_matrix = cp.asarray(nx.to_numpy_array(G))
        adj_matrix[adj_matrix > 0] = 1  # Convert unweighted graph to binary adjacency matrix

    results = [None] * seed_ensemble

    # Run HONE_worker_gpu in parallel using CuPy
    for seed in range(seed_ensemble):
        results[seed] = HONE_worker_gpu(adj_matrix, dim, iterations, tol, seed, dt, float(gamma))

    # Extract node positions and distance matrices from results
    ensemble_positions = [result[0] for result in results]
    distance_matrices = cp.array([result[1] for result in results])

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
