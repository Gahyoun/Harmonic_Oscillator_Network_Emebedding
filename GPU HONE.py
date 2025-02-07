import cupy as cp  # CuPy 사용
import numpy as np


def HONE(adj_matrix, dim=2, num_steps=1000, dt=0.01, gamma=0.1, seed=None):
    """
    Perform Harmonic Optimization using Molecular Dynamics (HONE) on GPU.
    - Utilizes CuPy for GPU acceleration.
    
    Parameters:
        adj_matrix (ndarray or cupy.ndarray): Adjacency matrix of the graph.
        dim (int): Dimensionality of the embedding space.
        num_steps (int): Number of simulation steps.
        dt (float): Time step for integration.
        gamma (float): Friction coefficient.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        positions_history (list of cupy.ndarray): History of node positions during optimization.
        energy_history (list of float): History of total system energy.
    """
    # Convert adjacency matrix to CuPy array if it's a NumPy array
    adj_matrix = cp.asarray(adj_matrix)

    if seed is not None:
        cp.random.seed(seed)

    num_nodes = adj_matrix.shape[0]

    # Initialize masses and positions
    masses = cp.sum(adj_matrix, axis=1)
    masses[masses == 0] = 1.0  # Handle isolated nodes

    positions = cp.random.rand(num_nodes, dim)
    velocities = cp.zeros((num_nodes, dim))

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
        kinetic_energy = 0.5 * cp.sum(masses[:, cp.newaxis] * velocities**2)
        potential_energy = cp.sum([
            0.5 * adj_matrix[i, j] * (cp.linalg.norm(positions[i] - positions[j]) - rest_lengths[(i, j)])**2
            for (i, j) in rest_lengths.keys()
        ])
        return float(kinetic_energy + potential_energy)

    for step in range(num_steps):
        forces = cp.zeros((num_nodes, dim))

        # Compute forces based on Hooke's Law
        for (i, j) in rest_lengths.keys():
            r_vec = positions[i] - positions[j]
            distance = cp.linalg.norm(r_vec)
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
        velocities += (forces / masses[:, cp.newaxis]) * (0.5 * dt)
        positions += velocities * dt

        # Recalculate forces after position update
        new_forces = cp.zeros((num_nodes, dim))

        for (i, j) in rest_lengths.keys():
            r_vec = positions[i] - positions[j]
            distance = cp.linalg.norm(r_vec)
            r_0 = rest_lengths[(i, j)]

            if distance > 1e-8:
                k_ij = adj_matrix[i, j]
                force_magnitude = -k_ij * (distance - r_0)
                force_vec = force_magnitude * (r_vec / distance)

                new_forces[i] += force_vec
                new_forces[j] -= force_vec

        # Apply friction to new forces
        new_forces -= gamma * velocities
        velocities += (new_forces / masses[:, cp.newaxis]) * (0.5 * dt)

        # Save energy and positions
        energy_history.append(compute_energy())
        positions_history.append(cp.asarray(positions.copy()))

    return positions_history, energy_history


def compute_distance_matrix(positions):
    """
    Compute the Euclidean distance matrix for final node positions on GPU.
    
    Parameters:
        positions (cupy.ndarray): Node positions.

    Returns:
        distance_matrix (cupy.ndarray): Euclidean distance matrix.
    """
    num_nodes = positions.shape[0]
    distance_matrix = cp.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = cp.linalg.norm(positions[i] - positions[j])
            distance_matrix[i, j] = distance_matrix[j, i] = distance

    return distance_matrix
