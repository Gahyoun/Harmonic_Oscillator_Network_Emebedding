import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor

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
    np.random.seed(seed)  
    num_nodes = adj_matrix.shape[0]  

    # ✅ 가중치 정규화 (발산 방지)
    max_weight = np.max(adj_matrix)
    if max_weight > 0:
        adj_matrix = adj_matrix / max_weight

    # ✅ 초기 위치 설정 (작은 무작위 값)
    positions = np.random.rand(num_nodes, dim) * 0.1  
    optimal_distances = np.copy(adj_matrix)  

    def compute_optimal_distances(positions):
        """ 네트워크 구조 기반 최적 거리 계산 """
        nonlocal optimal_distances
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:
                    k_ij = adj_matrix[i, j]  
                    r_ij = np.linalg.norm(positions[j] - positions[i])  
                    gradient = -k_ij * (r_ij - optimal_distances[i, j])  
                    optimal_distances[i, j] = r_ij - gradient / k_ij  
        return optimal_distances

    def compute_forces(positions, optimal_distances):
        """ 스프링 모델 기반 힘 계산 """
        forces = np.zeros_like(positions)  
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:
                    k_ij = adj_matrix[i, j]  
                    r_ij = positions[j] - positions[i]  
                    distance = np.linalg.norm(r_ij)  
                    unit_vector = r_ij / distance if distance != 0 else np.zeros_like(r_ij)

                    # ✅ 안정적인 복원력 계산
                    force_magnitude = -k_ij * (distance - optimal_distances[i, j])
                    forces[i] += force_magnitude * unit_vector
        return forces

    # 시뮬레이션 루프
    for _ in range(iterations):
        optimal_distances = compute_optimal_distances(positions)
        forces = compute_forces(positions, optimal_distances)

        # ✅ 위치 업데이트 (발산 방지를 위한 안정적인 스텝 크기)
        new_positions = positions - (forces / gamma) * dt  

        # ✅ 수렴 체크
        total_movement = np.sum(np.linalg.norm(new_positions - positions, axis=1))
        if total_movement < tol:
            break

        positions = new_positions

    # ✅ 최종 거리 계산
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)

    return positions, distances

def HONE(G, dim=2, iterations=100, seed_ensemble=100, tol=1e-4, dt=0.01, gamma=1):
    """
    Perform Harmonic Oscillator Network Embedding (HONE) for a given graph.

    Parameters:
        G (networkx.Graph): Input graph to be embedded.
        dim (int): Number of dimensions for the embedding space.
        iterations (int): Maximum number of iterations.
        seed_ensemble (int): Number of random initializations (seeds) for ensemble calculation.
        tol (float): Convergence tolerance for the total movement of positions.
        dt (float): Time step for the integration process.
        gamma (float): Damping coefficient for the overdamped dynamics.

    Returns:
        tuple:
            - ensemble_positions (list of np.ndarray): List of node positions for each ensemble (length: seed_ensemble).
            - distance_matrices (np.ndarray): Array of pairwise distance matrices for each ensemble (shape: seed_ensemble x num_nodes x num_nodes).
    """
    # ✅ 그래프의 가중치 정규화 (발산 방지)
    if nx.is_weighted(G):
        adj_matrix = np.asarray(nx.to_numpy_array(G, weight="weight"))
        max_weight = np.max(adj_matrix)
        if max_weight > 0:
            adj_matrix = adj_matrix / max_weight
    else:
        adj_matrix = np.asarray(nx.to_numpy_array(G))
        adj_matrix[adj_matrix > 0] = 1  

    results = [None] * seed_ensemble

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(HONE_worker, adj_matrix, dim, iterations, tol, seed, dt, float(gamma))  
            for seed in range(seed_ensemble)
        ]
        for i, future in enumerate(futures):
            results[i] = future.result()

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
