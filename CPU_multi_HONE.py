import numpy as np
from concurrent.futures import ProcessPoolExecutor


def HONE(adj_matrix, dim=2, num_steps=1000, dt=0.01, gamma=0.1, seed=None):
    """
    Perform Harmonic Optimization using Molecular Dynamics (HONE).
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

    positions_history = []
    energy_history = []

    def compute_energy():
        """Compute the total system energy (kinetic + potential)."""
        kinetic_energy = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
        potential_energy = sum(
            0.5 * adj_matrix[i, j] * (np.linalg.norm(positions[i] - positions[j]) - rest_lengths[(i, j)])**2
            for (i, j) in rest_lengths.keys()
        )
        return kinetic_energy + potential_energy

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

        # Save energy and positions
        energy_history.append(compute_energy())
        positions_history.append(positions.copy())

    return positions_history, energy_history


def compute_distance_matrix(positions):
    """
    Compute the Euclidean distance matrix for final node positions.
    """
    num_nodes = positions.shape[0]
    distance_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = np.linalg.norm(positions[i] - positions[j])
            distance_matrix[i, j] = distance_matrix[j, i] = distance

    return distance_matrix


def single_process_run(seed, adj_matrix, dim, num_steps, dt):
    """
    Wrapper function to run HONE for a single seed.
    """
    positions_history, _ = HONE(adj_matrix, dim, num_steps, dt, seed=seed)
    return positions_history[-1]  # Return final positions


def parallel_HONE(adj_matrix, dim=2, num_steps=1000, dt=0.01, seed_ensemble=10, max_workers=None):
    """
    Perform multiple independent runs of HONE using parallel processing.
    
    Parameters:
        adj_matrix (ndarray): Adjacency matrix of the graph.
        dim (int): Dimensionality of the embedding space.
        num_steps (int): Number of simulation steps.
        dt (float): Time step for integration.
        seed_ensemble (int): Number of independent runs.
        max_workers (int, optional): Maximum number of worker processes.

    Returns:
        results (list): List of final node positions from each run.
        distance_matrices (ndarray): Distance matrices for each embedding.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit HONE runs to the executor
        futures = [
            executor.submit(single_process_run, seed, adj_matrix, dim, num_steps, dt)
            for seed in range(seed_ensemble)
        ]
        results = [future.result() for future in futures]

    # Compute distance matrices for each embedding
    distance_matrices = np.array([compute_distance_matrix(result) for result in results])
    return results, distance_matrices


def HNI(distance_matrices):
    """
    Compute the Harmonic Network Inconsistency (HNI) value.
    """
    pairwise_variances = np.var(distance_matrices, axis=0)
    upper_tri_indices = np.triu_indices_from(pairwise_variances, k=1)
    upper_tri_variances = pairwise_variances[upper_tri_indices]

    hni_value = np.nanmean(upper_tri_variances) if not np.isnan(upper_tri_variances).all() else 0
    return hni_value
