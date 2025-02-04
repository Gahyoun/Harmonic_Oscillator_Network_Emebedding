import cupy as cp
import networkx as nx
from concurrent.futures import ThreadPoolExecutor

def HONE_worker_GPU(adj_matrix, dim, iterations, tol, seed, dt, gamma):
    """
    GPU-accelerated worker function for Harmonic Oscillator Network Embedding (HONE) 
    using CuPy in the overdamped limit.

    Parameters:
        adj_matrix (cp.ndarray): Adjacency matrix on GPU.
        dim (int): Number of dimensions for embedding space.
        iterations (int): Maximum number of iterations for convergence.
        tol (float): Convergence threshold based on total movement.
        seed (int): Random seed for reproducibility.
        dt (float): Time step for numerical integration.
        gamma (float): Damping coefficient for overdamped dynamics.

    Returns:
        tuple:
            - positions (cp.ndarray): Final positions of nodes in embedding space (GPU array).
            - distances (cp.ndarray): Pairwise node distances in final embedding (GPU array).
    """
    cp.random.seed(seed)  
    num_nodes = adj_matrix.shape[0]  

    # Normalize adjacency matrix weights
    total_weight = cp.sum(adj_matrix)
    if total_weight > 0:
        adj_matrix = adj_matrix / total_weight

    # Initialize positions with random small perturbations
    positions = cp.random.rand(num_nodes, dim)
    optimal_distances = cp.copy(adj_matrix)  

    def compute_optimal_distances(positions):
        """ Compute optimal distances between nodes based on network structure. """
        nonlocal optimal_distances
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:  
                    k_ij = adj_matrix[i, j]  
                    r_ij = cp.linalg.norm(positions[j] - positions[i])  
                    gradient = -k_ij * (r_ij - optimal_distances[i, j])  
                    optimal_distances[i, j] = r_ij - gradient / k_ij  
        return optimal_distances

    def compute_forces(positions, optimal_distances):
        """ Compute restoring forces acting on each node using CuPy for GPU acceleration. """
        forces = cp.zeros_like(positions)  
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:  
                    k_ij = adj_matrix[i, j]  
                    r_ij = positions[j] - positions[i]  
                    distance = cp.linalg.norm(r_ij)  

                    unit_vector = r_ij / distance if distance > 1e-6 else cp.zeros_like(r_ij)
                    force_magnitude = -k_ij * (distance - optimal_distances[i, j])
                    forces[i] += force_magnitude * unit_vector  
        return forces

    # Iterative simulation loop for convergence
    for _ in range(iterations):
        optimal_distances = compute_optimal_distances(positions)
        forces = compute_forces(positions, optimal_distances)

        # Update positions using overdamped dynamics
        new_positions = positions - (forces / gamma) * dt  

        # Check convergence based on total movement
        total_movement = cp.sum(cp.linalg.norm(new_positions - positions, axis=1))
        if total_movement < tol:
            break

        positions = new_positions  

    # Compute final pairwise distances between nodes
    distances = cp.linalg.norm(positions[:, None] - positions[None, :], axis=2)

    return positions, distances

def HONE_GPU(G, dim=2, iterations=100, seed_ensemble=100, tol=1e-4, dt=0.01, gamma=1):
    """
    GPU-accelerated version of HONE using CuPy for a given graph.

    Parameters:
        G (networkx.Graph): Input network graph.
        dim (int): Number of dimensions for embedding space.
        iterations (int): Maximum number of iterations for the embedding process.
        seed_ensemble (int): Number of random initializations (seeds) for ensemble calculation.
        tol (float): Convergence tolerance based on total movement.
        dt (float): Time step for numerical integration.
        gamma (float): Damping coefficient for overdamped dynamics.

    Returns:
        tuple:
            - ensemble_positions (list of cp.ndarray): Node positions for each ensemble.
            - distance_matrices (cp.ndarray): Pairwise distance matrices (shape: seed_ensemble x num_nodes x num_nodes).
    """
    # Convert graph to weighted adjacency matrix (stored in GPU memory)
    if nx.is_weighted(G):
        adj_matrix = cp.asarray(nx.to_numpy_array(G, weight="weight"))
    else:
        adj_matrix = cp.asarray(nx.to_numpy_array(G))
        adj_matrix[adj_matrix > 0] = 1  

    # Normalize adjacency matrix by total weight sum
    total_weight = cp.sum(adj_matrix)
    if total_weight > 0:
        adj_matrix = adj_matrix / total_weight  

    results = [None] * seed_ensemble

    # Run HONE_worker_GPU in parallel using multiple seeds
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(HONE_worker_GPU, adj_matrix, dim, iterations, tol, seed, dt, float(gamma))
            for seed in range(seed_ensemble)
        ]
        for i, future in enumerate(futures):
            results[i] = future.result()

    # Extract node positions and distance matrices from results
    ensemble_positions = [result[0] for result in results]
    distance_matrices = cp.array([result[1] for result in results])

    return ensemble_positions, distance_matrices

def HNI_GPU(distance_matrices):
    """
    Calculate the Harmonic Network Inconsistency (HNI) using GPU.

    Parameters:
        distance_matrices (cp.ndarray): Array of pairwise distance matrices for each ensemble 
                                        (shape: seed_ensemble x num_nodes x num_nodes).

    Returns:
        float: Average variance of pairwise distances across ensembles.
    """
    # Compute variance using CuPy (GPU)
    pairwise_variances = cp.var(distance_matrices, axis=0)  

    # Extract the upper triangular part of the variance matrix
    upper_tri_indices = cp.triu_indices_from(pairwise_variances, k=1)
    upper_tri_variances = pairwise_variances[upper_tri_indices]

    # Return the average variance as the HNI value
    return cp.mean(upper_tri_variances).item()
