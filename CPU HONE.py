import numpy as np
from concurrent.futures import ProcessPoolExecutor

def HONE(adj_matrix, dim=2, num_steps=1000, learning_rate=0.01, seed=None):
    """
    Harmonic Oscillator Network Embedding (HONE)
    """
    if seed is not None:
        np.random.seed(seed)

    num_nodes = adj_matrix.shape[0]

    # Initialize random positions
    positions = {node: np.random.rand(dim) for node in range(num_nodes)}

    # Compute rest lengths (equilibrium distances) for edges (inverse of weights)
    rest_lengths = {
        (i, j): 1 / adj_matrix[i, j]
        for i in range(num_nodes) for j in range(num_nodes)
        if adj_matrix[i, j] > 0 and i != j  # 자기 자신 제외
    }

    # Gradient optimization loop
    for _ in range(num_steps):
        new_positions = {}
        for node in range(num_nodes):
            gradient = np.zeros(dim)
            for neighbor in range(num_nodes):
                if node != neighbor and adj_matrix[node, neighbor] > 0:  # 자기 자신 제외
                    distance = np.linalg.norm(positions[node] - positions[neighbor])  # 1e-6 제거
                    r_0 = rest_lengths[(node, neighbor)]
                    diff = (distance - r_0) / distance if distance > 0 else 0  # Zero division 방지
                    gradient += diff * (positions[node] - positions[neighbor])

            # Gradient descent update
            new_positions[node] = positions[node] - learning_rate * gradient

        positions = new_positions  # Apply updates

    return positions

def compute_distance_matrix(positions):
    """
    Compute the Euclidean distance matrix for final node positions.
    """
    nodes = list(positions.keys())
    num_nodes = len(nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = np.linalg.norm(positions[nodes[i]] - positions[nodes[j]])
            distance_matrix[i, j] = distance_matrix[j, i] = distance

    return distance_matrix

def parallel_HONE(adj_matrix, dim=2, num_steps=1000, learning_rate=0.01, seed_ensemble=10):
    """
    Perform multiple independent runs of HONE in parallel using multiprocessing.
    """
    results = [None] * seed_ensemble

    # Run HONE in parallel using multiple processes
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(HONE, adj_matrix, dim, num_steps, learning_rate, seed)
            for seed in range(seed_ensemble)
        ]
        for i, future in enumerate(futures):
            results[i] = future.result()

    # Extract node positions and compute distance matrices
    ensemble_positions = results
    distance_matrices = np.array([compute_distance_matrix(result) for result in results])

    return ensemble_positions, distance_matrices

def HNI(distance_matrices):
    """
    Compute the Harmonic Network Inconsistency (HNI) value.
    """
    # Compute variance for each pair of nodes across different embeddings
    pairwise_variances = np.var(distance_matrices, axis=0)

    # Extract upper triangular part (excluding diagonal) to avoid redundancy
    upper_tri_indices = np.triu_indices_from(pairwise_variances, k=1)
    upper_tri_variances = pairwise_variances[upper_tri_indices]

    # Compute mean variance (HNI), handling NaN cases
    hni_value = np.nanmean(upper_tri_variances) if not np.isnan(upper_tri_variances).all() else 0
    return hni_value
    
