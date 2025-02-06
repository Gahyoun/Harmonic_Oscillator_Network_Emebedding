import cupy as cp
from concurrent.futures import ProcessPoolExecutor

def HONE(adj_matrix, dim=2, num_steps=1000, learning_rate=0.01, seed=None):
    """
    Harmonic Oscillator Network Embedding (HONE) - GPU Version using CuPy.
    """
    if seed is not None:
        cp.random.seed(seed)

    num_nodes = adj_matrix.shape[0]

    # Initialize random positions (GPU array)
    positions = cp.random.rand(num_nodes, dim)

    # Compute rest lengths (inverse of weights) for edges (ignoring self-loops)
    rest_lengths = cp.zeros((num_nodes, num_nodes))
    mask = (adj_matrix > 0) & (cp.arange(num_nodes)[:, None] != cp.arange(num_nodes))  # Ignore self-loops
    rest_lengths[mask] = 1 / adj_matrix[mask]

    # Gradient optimization loop
    for _ in range(num_steps):
        gradients = cp.zeros((num_nodes, dim))

        for node in range(num_nodes):
            neighbors = cp.where(adj_matrix[node] > 0)[0]
            if neighbors.size > 0:
                distances = cp.linalg.norm(positions[node] - positions[neighbors], axis=1)
                r_0 = rest_lengths[node, neighbors]

                # Avoid zero division
                diff = cp.where(distances > 0, (distances - r_0) / distances, 0).reshape(-1, 1)

                gradients[node] = cp.sum(diff * (positions[node] - positions[neighbors]), axis=0)

        # Update positions using gradient descent
        positions -= learning_rate * gradients

    return positions

def compute_distance_matrix(positions):
    """
    Compute the Euclidean distance matrix for final node positions using CuPy.
    """
    num_nodes = positions.shape[0]
    distance_matrix = cp.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = cp.linalg.norm(positions[i] - positions[j])
            distance_matrix[i, j] = distance_matrix[j, i] = distance

    return distance_matrix

def parallel_HONE(adj_matrix, dim=2, num_steps=1000, learning_rate=0.01, seed_ensemble=10):
    """
    Perform multiple independent runs of HONE in parallel using multiprocessing (CuPy-based GPU).
    """
    results = [None] * seed_ensemble

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(HONE, adj_matrix, dim, num_steps, learning_rate, seed)
            for seed in range(seed_ensemble)
        ]
        for i, future in enumerate(futures):
            results[i] = future.result()

    # Compute distance matrices for each embedding
    ensemble_positions = results
    distance_matrices = cp.array([compute_distance_matrix(result) for result in results])

    return ensemble_positions, distance_matrices

def HNI(distance_matrices):
    """
    Compute the Harmonic Network Inconsistency (HNI) value using GPU.
    """
    # Compute variance for each pair of nodes across different embeddings
    pairwise_variances = cp.var(distance_matrices, axis=0)

    # Extract upper triangular part (excluding diagonal)
    upper_tri_indices = cp.triu_indices_from(pairwise_variances, k=1)
    upper_tri_variances = pairwise_variances[upper_tri_indices]

    # Compute mean variance (HNI), handling NaN cases
    hni_value = cp.nanmean(upper_tri_variances) if not cp.isnan(upper_tri_variances).all() else 0
    return hni_value
