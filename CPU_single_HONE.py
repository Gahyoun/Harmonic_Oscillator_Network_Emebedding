import numpy as np



def HONE(adj_matrix, dim=2, num_steps=1000, dt=0.01, gamma=0.1, seed=None):
    """
    Perform Harmonic Optimization using Molecular Dynamics (HONE).
    - Verlet Integration applied for numerical stability.
    - Langevin Dynamics included for friction.

    Parameters:
        adj_matrix (ndarray): Adjacency matrix of the graph.
        dim (int): Dimensionality of the embedding space.
        num_steps (int): Number of simulation steps.
        dt (float): Time step for integration.
        gamma (float): Friction coefficient.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        positions (ndarray): Final node positions after optimization.
    """
    if seed is not None:
        np.random.seed(seed)

    num_nodes = adj_matrix.shape[0]

    # Initialize masses and positions
    masses = np.sum(adj_matrix, axis=1)
    masses[masses == 0] = 1.0  # Handle isolated nodes

    positions = np.random.rand(num_nodes, dim)
    velocities = np.zeros((num_nodes, dim))

    # Define equilibrium lengths for edges
    rest_lengths = {
        (i, j): 1 / adj_matrix[i, j] if adj_matrix[i, j] > 0 else 0
        for i in range(num_nodes) for j in range(i + 1, num_nodes)
        if adj_matrix[i, j] > 0
    }

    for step in range(num_steps):
        forces = np.zeros((num_nodes, dim))

        # Compute forces based on Hooke's Law
        for (i, j) in rest_lengths.keys():
            r_vec = positions[i] - positions[j]
            distance = np.linalg.norm(r_vec)
            r_0 = rest_lengths[(i, j)]

            if distance > 1e-8:  # Prevent division by zero
                k_ij = adj_matrix[i, j]
                force_magnitude = -k_ij * (distance - r_0)
                force_vec = force_magnitude * (r_vec / distance)

                forces[i] += force_vec
                forces[j] -= force_vec

        # Apply friction
        forces -= gamma * velocities

        # Verlet Integration
        velocities += (forces / masses[:, np.newaxis]) * (0.5 * dt)
        positions += velocities * dt

        # Recalculate forces after position update
        new_forces = np.zeros((num_nodes, dim))

        for (i, j) in rest_lengths.keys():
            r_vec = positions[i] - positions[j]
            distance = np.linalg.norm(r_vec)
            r_0 = rest_lengths[(i, j)]

            if distance > 1e-8:
                k_ij = adj_matrix[i, j]
                force_magnitude = -k_ij * (distance - r_0)
                force_vec = force_magnitude * (r_vec / distance)

                new_forces[i] += force_vec
                new_forces[j] -= force_vec

        # Apply friction to new forces
        new_forces -= gamma * velocities
        velocities += (new_forces / masses[:, np.newaxis]) * (0.5 * dt)

    return positions 


def compute_distance_matrix(positions):
    """
    Compute the Euclidean distance matrix for final node positions.
    
    Parameters:
        positions (ndarray): Node positions.

    Returns:
        distance_matrix (ndarray): Euclidean distance matrix.
    """
    num_nodes = positions.shape[0]
    distance_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = np.linalg.norm(positions[i] - positions[j])
            distance_matrix[i, j] = distance_matrix[j, i] = distance

    return distance_matrix


def single_process_HONE(adj_matrix, dim=2, num_steps=1000, dt=0.01, seed_ensemble=10):
    """
    Perform multiple independent runs of HONE sequentially (single process).
    
    Parameters:
        adj_matrix (ndarray): Adjacency matrix of the graph.
        dim (int): Dimensionality of the embedding space.
        num_steps (int): Number of simulation steps.
        dt (float): Time step for integration.
        seed_ensemble (int): Number of independent runs.

    Returns:
        results (list): List of position histories from each run.
        distance_matrices (ndarray): Distance matrices for each embedding.
    """
    results = []

    for seed in range(seed_ensemble):
        positions_history, _ = HONE(adj_matrix, dim, num_steps, dt, seed=seed)
        results.append(positions_history[-1])  # Save final positions

    distance_matrices = np.array([compute_distance_matrix(result) for result in results])
    return results, distance_matrices


def HNI(distance_matrices):
    """
    Compute the Harmonic Network Inconsistency (HNI) value.
    
    Parameters:
        distance_matrices (ndarray): Array of distance matrices from different embeddings.

    Returns:
        hni_value (float): Mean pairwise variance across embeddings.
    """
    pairwise_variances = np.var(distance_matrices, axis=0)
    upper_tri_indices = np.triu_indices_from(pairwise_variances, k=1)
    upper_tri_variances = pairwise_variances[upper_tri_indices]

    hni_value = np.nanmean(upper_tri_variances) if not np.isnan(upper_tri_variances).all() else 0
    return hni_value

