import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor

def HONE_worker(adj_matrix, dim, iterations, tol, seed, dt, gamma):
    """
    Worker function for Harmonic Oscillator Network Embedding (HONE) in the overdamped limit.

    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix where weights represent spring constants.
        dim (int): Number of dimensions for embedding space.
        iterations (int): Maximum number of iterations for convergence.
        tol (float): Convergence threshold based on total movement.
        seed (int): Random seed for reproducibility.
        dt (float): Time step for numerical integration.
        gamma (float): Damping coefficient for overdamped dynamics.

    Returns:
        tuple:
            - positions (np.ndarray): Final positions of nodes in embedding space (shape: num_nodes x dim).
            - distances (np.ndarray): Pairwise node distances in final embedding (shape: num_nodes x num_nodes).
    """
    np.random.seed(seed)  # Set random seed for reproducibility
    num_nodes = adj_matrix.shape[0]  # Number of nodes in the network

    # Normalize adjacency matrix weights to avoid numerical instability
    max_weight = np.max(adj_matrix)
    if max_weight > 0:
        adj_matrix = adj_matrix / max_weight

    # Initialize node positions with small perturbations
    positions = np.random.rand(num_nodes, dim)
    optimal_distances = np.copy(adj_matrix)  # Initialize optimal distances using adjacency weights

    def compute_optimal_distances(positions):
        """ Compute the optimal distances between nodes based on network structure. """
        nonlocal optimal_distances
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:  # Only process connected nodes
                    k_ij = adj_matrix[i, j]  # Spring constant (edge weight)
                    r_ij = np.linalg.norm(positions[j] - positions[i])  # Current node distance
                    gradient = -k_ij * (r_ij - optimal_distances[i, j])  # Gradient of energy function
                    optimal_distances[i, j] = r_ij - gradient / k_ij  # Update optimal distance
        return optimal_distances

    def compute_forces(positions, optimal_distances):
        """ Compute restoring forces acting on each node based on the spring model. """
        forces = np.zeros_like(positions)  # Initialize forces array
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:  # Only consider connected nodes
                    k_ij = adj_matrix[i, j]  # Spring constant
                    r_ij = positions[j] - positions[i]  # Vector displacement
                    distance = np.linalg.norm(r_ij)  # Scalar distance

                    # Avoid division by zero when calculating unit vector
                    unit_vector = r_ij / distance if distance > 1e-6 else np.zeros_like(r_ij)

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
        total_movement = np.sum(np.linalg.norm(new_positions - positions, axis=1))
        if total_movement < tol:
            break

        positions = new_positions  # Update positions for next iteration

    # Compute final pairwise distances between nodes
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)

    return positions, distances

def HONE(G, dim=2, iterations=100, seed_ensemble=100, tol=1e-4, dt=0.01, gamma=1):
    """
    Perform Harmonic Oscillator Network Embedding (HONE) for a given graph.

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
            - ensemble_positions (list of np.ndarray): Node positions for each ensemble (length: seed_ensemble).
            - distance_matrices (np.ndarray): Pairwise distance matrices for each ensemble (shape: seed_ensemble x num_nodes x num_nodes).
    """
    # Convert graph to weighted adjacency matrix
    if nx.is_weighted(G):
        adj_matrix = np.asarray(nx.to_numpy_array(G, weight="weight"))
        max_weight = np.max(adj_matrix)
        if max_weight > 0:
            adj_matrix = adj_matrix / max_weight  # Normalize weights
    else:
        adj_matrix = np.asarray(nx.to_numpy_array(G))
        adj_matrix[adj_matrix > 0] = 1  # Convert unweighted graph to binary adjacency matrix

    results = [None] * seed_ensemble

    # Run HONE_worker in parallel for multiple random seeds
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(HONE_worker, adj_matrix, dim, iterations, tol, seed, dt, float(gamma))
            for seed in range(seed_ensemble)
        ]
        for i, future in enumerate(futures):
            results[i] = future.result()

    # Extract node positions and distance matrices from results
    ensemble_positions = [result[0] for result in results]
    distance_matrices = np.array([result[1] for result in results])

    return ensemble_positions, distance_matrices

def HNI(distance_matrices):
    """
    Calculate the Harmonic Network Inconsistency (HNI) to quantify variance across ensembles.

    Parameters:
        distance_matrices (np.ndarray): Array of pairwise distance matrices for each ensemble 
                                        (shape: seed_ensemble x num_nodes x num_nodes).

    Returns:
        float: Average variance of pairwise distances across ensembles.
    """
    # Compute variance of distances for each pair of nodes across ensembles
    pairwise_variances = np.var(distance_matrices, axis=0)  # Shape: (num_nodes x num_nodes)

    # Extract the upper triangular part of the variance matrix (excluding the diagonal)
    upper_tri_indices = np.triu_indices_from(pairwise_variances, k=1)
    upper_tri_variances = pairwise_variances[upper_tri_indices]

    # Return the average variance as the HNI value
    return np.mean(upper_tri_variances)
