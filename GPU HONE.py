import numpy as np
import cupy as cp
import networkx as nx

def HONE_worker(adj_matrix, dim, iterations, tol, seed, dt, gamma, stream):
    """
    Worker function for Harmonic Oscillator Network Embedding (HONE) using overdamped dynamics with CUDA streams.

    Parameters:
        adj_matrix (cp.ndarray): Adjacency matrix of the network, stored on the GPU.
        dim (int): Number of dimensions for the embedding space.
        iterations (int): Maximum number of iterations to run the embedding process.
        tol (float): Convergence tolerance for the total movement of positions.
        seed (int): Random seed for initializing positions.
        dt (float): Time step for the integration process.
        gamma (float): Damping coefficient for the overdamped dynamics.
        stream (cp.cuda.Stream): CUDA stream to allow asynchronous GPU operations.

    Returns:
        tuple:
            - positions (np.ndarray): Final positions of nodes in the embedding space (shape: num_nodes x dim).
            - distances (np.ndarray): Pairwise distances between nodes in the final embedding (shape: num_nodes x num_nodes).
    """
    with stream:  # Activate the CUDA stream
        # Set the random seed for reproducibility on the GPU
        cp.random.seed(seed)

        # Initialize positions randomly in the embedding space
        positions = cp.random.rand(adj_matrix.shape[0], dim)

        def calculate_forces(positions):
            """
            Calculate forces based on the harmonic oscillator model using GPU computations.

            Parameters:
                positions (cp.ndarray): Current positions of nodes in the embedding space.

            Returns:
                cp.ndarray: Forces acting on each node (shape: num_nodes x dim).
            """
            forces = cp.zeros_like(positions)
            for i in range(len(positions)):
                # Calculate displacement vectors from node i to all other nodes
                delta = positions - positions[i]
                # Mask to avoid division by zero (self-loops)
                mask = cp.arange(len(positions)) != i
                # Compute forces based on adjacency matrix and displacement vectors
                forces[i] = cp.sum(adj_matrix[i, mask][:, None] * delta[mask], axis=0)
            return forces

        # Iterative integration process
        for _ in range(iterations):
            # Calculate forces for the current positions
            forces = calculate_forces(positions)
            # Update positions based on overdamped dynamics
            new_positions = positions + (forces / gamma) * dt
            # Calculate total movement to check for convergence
            total_movement = cp.sum(cp.linalg.norm(new_positions - positions, axis=1))
            if total_movement < tol:  # Convergence condition
                break
            # Update positions for the next iteration
            positions = new_positions

        # Calculate the pairwise distances in the final embedding
        distances = cp.linalg.norm(positions[:, None] - positions[None, :], axis=2)
        # Convert GPU results to CPU-friendly numpy arrays
        return cp.asnumpy(positions), cp.asnumpy(distances)

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
