import cupy as cp
import networkx as nx

def HONE_worker_gpu(adj_matrix, dim, iterations, tol, seed, dt, gamma):
    """
    GPU-accelerated worker function for Harmonic Oscillator Network Embedding (HONE).

    Parameters:
        adj_matrix (cp.ndarray): Adjacency matrix stored in GPU memory.
        dim (int): Number of embedding dimensions.
        iterations (int): Maximum number of iterations.
        tol (float): Convergence threshold.
        seed (int): Random seed.
        dt (float): Time step.
        gamma (float): Damping coefficient.

    Returns:
        tuple:
            - positions (cp.ndarray): Final positions of nodes in the embedding space (shape: num_nodes x dim).
            - distances (cp.ndarray): Pairwise node distances in the final embedding (shape: num_nodes x num_nodes).
    """
    cp.random.seed(seed)  # Random seed for reproducibility
    num_nodes = adj_matrix.shape[0]  # Number of nodes

    # Initialize positions randomly on GPU
    positions = cp.random.rand(num_nodes, dim) * 0.1
    optimal_distances = cp.copy(adj_matrix)  # Initialize optimal distances

    def compute_optimal_distances(positions):
        """
        Compute optimal distances based on the gradient of the energy landscape.
        """
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
        """
        Calculate forces on each node based on the harmonic oscillator model.
        """
        forces = cp.zeros_like(positions)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:
                    k_ij = adj_matrix[i, j]
                    r_ij = positions[j] - positions[i]
                    distance = cp.linalg.norm(r_ij)
                    unit_vector = r_ij / distance if distance != 0 else cp.zeros_like(r_ij)

                    # Compute restoring force
                    force_magnitude = -k_ij * (distance - optimal_distances[i, j])
                    forces[i] += force_magnitude * unit_vector
        return forces

    # Simulation loop
    for _ in range(iterations):
        optimal_distances = compute_optimal_distances(positions)
        forces = compute_forces(positions, optimal_distances)

        # Update positions with overdamped dynamics
        new_positions = positions - (forces / float(gamma)) * dt

        # Check convergence
        total_movement = cp.sum(cp.linalg.norm(new_positions - positions, axis=1))
        if total_movement < tol:
            break

        positions = new_positions

    # Compute pairwise distances
    distances = cp.linalg.norm(positions[:, None] - positions[None, :], axis=2)

    return positions, distances

def HONE_gpu(G, dim=2, iterations=100, seed_ensemble=100, tol=1e-4, dt=0.01, gamma=1.0):
    """
    GPU-based Harmonic Oscillator Network Embedding (HONE) for a given graph.

    Parameters:
        G (networkx.Graph): Input graph.
        dim (int): Number of embedding dimensions.
        iterations (int): Maximum number of iterations.
        seed_ensemble (int): Number of random initializations.
        tol (float): Convergence tolerance.
        dt (float): Time step.
        gamma (float): Damping coefficient.

    Returns:
        tuple:
            - ensemble_positions (list of cp.ndarray): List of node positions for each ensemble.
            - distance_matrices (cp.ndarray): Array of pairwise distance matrices (shape: seed_ensemble x num_nodes x num_nodes).
    """
    # Convert adjacency matrix to CuPy array
    if nx.is_weighted(G):
        adj_matrix = cp.asarray(nx.to_numpy_array(G, weight="weight"))
    else:
        adj_matrix = cp.asarray(nx.to_numpy_array(G))
        adj_matrix[adj_matrix > 0] = 1  

    # GPU execution
    results = [HONE_worker_gpu(adj_matrix, dim, iterations, tol, seed, dt, float(gamma)) for seed in range(seed_ensemble)]

    # Extract node positions and distance matrices
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
