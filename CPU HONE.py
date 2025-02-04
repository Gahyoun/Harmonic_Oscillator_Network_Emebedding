import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor

def HONE_worker(adj_matrix, dim, iterations, tol, seed, dt, gamma):
    """
    Worker function for Harmonic Oscillator Network Embedding (HONE) in the overdamped limit.

    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix (weights as normalized spring constants).
        dim (int): Number of embedding dimensions.
        iterations (int): Maximum number of iterations.
        tol (float): Convergence threshold.
        seed (int): Random seed.
        dt (float): Time step.
        gamma (float): Damping coefficient for the overdamped dynamics.

    Returns:
        tuple:
            - positions (np.ndarray): Final positions of nodes in the embedding space.
            - distances (np.ndarray): Pairwise node distances in the final embedding.
    """
    np.random.seed(seed)  
    num_nodes = adj_matrix.shape[0]  

    # ✅ Normalize adjacency matrix weights so that total weight sum = 1
    total_weight = np.sum(adj_matrix)
    if total_weight > 0:
        adj_matrix /= total_weight  

    # ✅ Step 1: Initialize positions and velocities using Normal Distribution N(0, 0.5)
    positions = np.random.normal(loc=0, scale=1, size=(num_nodes, dim))
    velocities = np.random.normal(loc=0, scale=1, size=(num_nodes, dim))

    # ✅ Step 2: Move one step without damping
    positions += velocities * dt  

    # ✅ Step 3: Compute initial distances and determine optimal distances
    initial_distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    optimal_distances = np.ones_like(adj_matrix)  # Assume initial optimal distance = 1

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] > 0:
                w_ij = adj_matrix[i, j]  # Normalized weight
                r_ij = positions[j] - positions[i]
                distance = np.linalg.norm(r_ij)
                energy_gradient = -w_ij * (distance - 1)  # Energy gradient dE/d(distance)
                optimal_distances[i, j] = distance - energy_gradient / w_ij  # Compute and fix optimal distance

    # ✅ Step 4: Dynamics using fixed optimal distances
    def compute_forces(positions):
        """ Compute forces based on Hooke's Law with attraction and repulsion. """
        forces = np.zeros_like(positions)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:
                    k_ij = adj_matrix[i, j]  # Normalized spring constant
                    r_ij = positions[j] - positions[i]
                    distance = np.linalg.norm(r_ij)

                    if distance > 1e-6:
                        unit_vector = r_ij / distance
                    else:
                        unit_vector = np.zeros_like(r_ij)

                    # ✅ Attraction: If distance > optimal distance, apply attractive force
                    # ✅ Repulsion: If distance < optimal distance, apply repulsive force
                    if distance > optimal_distances[i, j]:  # Attraction
                        force_magnitude = - k_ij * (distance - optimal_distances[i, j])  
                    else:  # Repulsion
                        force_magnitude = + k_ij * (distance - optimal_distances[i, j])  

                    forces[i] += force_magnitude * unit_vector  

        return forces

    # ✅ Step 5: Simulation loop using overdamped dynamics
    for _ in range(iterations):
        forces = compute_forces(positions)

        # ✅ Update positions using overdamped limit
        new_positions = positions - (forces / gamma) * dt  

        # ✅ Check convergence
        total_movement = np.sum(np.linalg.norm(new_positions - positions, axis=1))
        if total_movement < tol:
            break

        positions = new_positions  

    # ✅ Step 6: Compute final pairwise distances
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)

    return positions, distances

def HONE(G, dim=2, iterations=100, seed_ensemble=10, tol=1e-4, dt=1, gamma=0.011):
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
    # ✅ Convert graph to weighted adjacency matrix
    if nx.is_weighted(G):
        adj_matrix = np.asarray(nx.to_numpy_array(G, weight="weight"))
    else:
        adj_matrix = np.asarray(nx.to_numpy_array(G))
        adj_matrix[adj_matrix > 0] = 1

    results = [None] * seed_ensemble

    # ✅ Run simulations in parallel using different seeds (0 to seed_ensemble-1)
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(HONE_worker, adj_matrix, dim, iterations, tol, seed, dt, float(gamma))  
            for seed in range(0, seed_ensemble)  # ✅ Ensure seed starts from 0 to (seed_ensemble-1)
        ]
        for i, future in enumerate(futures):
            results[i] = future.result()

    # ✅ Extract results
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
    pairwise_variances = np.var(distance_matrices, axis=0)
    # Extract the upper triangular part of the variance matrix (excluding the diagonal)
    upper_tri_indices = np.triu_indices_from(pairwise_variances, k=1)
    upper_tri_variances = pairwise_variances[upper_tri_indices]
    return np.mean(upper_tri_variances)
