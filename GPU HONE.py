import numpy as np
import cupy as cp
import networkx as nx

def HONE_worker(adj_matrix, dim, iterations, tol, seed, dt, gamma):
    """
    Worker function for Harmonic Oscillator Network Embedding (HONE) in the overdamped limit.

    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix (weights as spring constants).
        dim (int): Number of embedding dimensions.
        iterations (int): Maximum number of iterations.
        tol (float): Convergence threshold.
        seed (int): Random seed.
        dt (float): Time step.
        gamma (float): Damping coefficient for the overdamped dynamics.

    Returns:
        tuple:
            - positions (np.ndarray): Final positions of nodes in the embedding space (shape: num_nodes x dim).
            - distances (np.ndarray): Pairwise node distances in the final embedding (shape: num_nodes x num_nodes).
    """
    np.random.seed(seed)  # Set the random seed for reproducibility
    num_nodes = adj_matrix.shape[0]  # Number of nodes in the graph

    # Initialize positions randomly in the embedding space with small perturbations
    positions = np.random.rand(num_nodes, dim) * 0.1
    optimal_distances = np.copy(adj_matrix)  # Initialize optimal distances as adjacency weights

    def compute_optimal_distances(positions):
        """
        Compute optimal distances based on the gradient of the energy landscape.
        """
        nonlocal optimal_distances
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:  # Only consider connected nodes
                    k_ij = adj_matrix[i, j]  # Spring constant (weight of the edge)
                    r_ij = np.linalg.norm(positions[j] - positions[i])  # Current distance
                    gradient = -k_ij * (r_ij - optimal_distances[i, j])  # Gradient of the energy
                    optimal_distances[i, j] = r_ij - gradient / k_ij  # Update optimal distance
        return optimal_distances

    def compute_forces(positions, optimal_distances):
        """
        Calculate forces acting on each node based on the harmonic oscillator model.
        """
        forces = np.zeros_like(positions)  # Initialize forces
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:  # Only consider connected nodes
                    k_ij = adj_matrix[i, j]  # Spring constant
                    r_ij = positions[j] - positions[i]  # Vector distance
                    distance = np.linalg.norm(r_ij)  # Scalar distance
                    unit_vector = r_ij / distance if distance != 0 else np.zeros_like(r_ij)

                    # Compute restoring force based on the optimal distance
                    force_magnitude = -k_ij * (distance - optimal_distances[i, j])
                    forces[i] += force_magnitude * unit_vector
        return forces

    # Simulation loop
    for _ in range(iterations):
        # Step 1: Compute optimal distances
        optimal_distances = compute_optimal_distances(positions)

        # Step 2: Compute forces
        forces = compute_forces(positions, optimal_distances)

        # Step 3: Update positions using overdamped dynamics
        new_positions = positions - (forces / gamma) * dt

        # Step 4: Check convergence
        total_movement = np.sum(np.linalg.norm(new_positions - positions, axis=1))
        if total_movement < tol:  # Stop if total movement is below the threshold
            break

        # Update positions for the next iteration
        positions = new_positions

    # Calculate the pairwise distances between final positions
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)

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
