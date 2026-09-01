"""
Synthetic Dataset Generator for IVF Adaptive Search Experiments
Generates datasets with controlled density distributions in R^128
"""

import numpy as np

def random_directions(n, dim=128):
    """Generate n random unit vectors in R^dim"""
    vecs = np.random.randn(n, dim)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

def generate_two_densities(N_db=20000, N_q=200, dim=128, seed=None):
    """
    Generate two-density dataset:
    - 70% dense (small norm ~0.8)
    - 30% sparse (large norm ~3.5)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_dense = int(0.7 * N_db)
    n_sparse = N_db - n_dense
    
    # Database vectors
    db_dense = random_directions(n_dense, dim) * (0.8 + np.random.randn(n_dense, 1) * 0.05)
    db_sparse = random_directions(n_sparse, dim) * (3.5 + np.random.randn(n_sparse, 1) * 0.05)
    db = np.vstack([db_dense, db_sparse])
    
    # Query vectors (same distribution)
    q_dense = random_directions(int(0.7 * N_q), dim) * (0.8 + np.random.randn(int(0.7 * N_q), 1) * 0.05)
    q_sparse = random_directions(int(0.3 * N_q), dim) * (3.5 + np.random.randn(int(0.3 * N_q), 1) * 0.05)
    queries = np.vstack([q_dense, q_sparse])
    
    np.random.shuffle(db)
    np.random.shuffle(queries)
    
    return db, queries

def generate_three_densities(N_db=20000, N_q=200, dim=128, seed=None):
    """
    Generate three-density dataset:
    - 50% dense (norm ~0.8)
    - 30% medium (norm ~1.8)
    - 20% sparse (norm ~3.5)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n1 = int(0.5 * N_db)
    n2 = int(0.3 * N_db)
    n3 = N_db - n1 - n2
    
    # Database vectors
    db1 = random_directions(n1, dim) * (0.8 + np.random.randn(n1, 1) * 0.05)
    db2 = random_directions(n2, dim) * (1.8 + np.random.randn(n2, 1) * 0.05)
    db3 = random_directions(n3, dim) * (3.5 + np.random.randn(n3, 1) * 0.05)
    db = np.vstack([db1, db2, db3])
    
    # Query vectors
    q1 = random_directions(int(0.5 * N_q), dim) * (0.8 + np.random.randn(int(0.5 * N_q), 1) * 0.05)
    q2 = random_directions(int(0.3 * N_q), dim) * (1.8 + np.random.randn(int(0.3 * N_q), 1) * 0.05)
    q3 = random_directions(int(0.2 * N_q), dim) * (3.5 + np.random.randn(int(0.2 * N_q), 1) * 0.05)
    queries = np.vstack([q1, q2, q3])
    
    np.random.shuffle(db)
    np.random.shuffle(queries)
    
    return db, queries

def generate_pareto_tails(N_db=20000, N_q=200, dim=128, seed=None):
    """
    Generate dataset with Pareto-distributed norms (heavy-tailed)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Pareto distribution for norms
    norms_db = (np.random.pareto(2.0, N_db) + 1) * 0.8
    norms_q = (np.random.pareto(2.0, N_q) + 1) * 0.8
    
    # Random directions scaled by norms
    db = random_directions(N_db, dim) * norms_db.reshape(-1, 1)
    queries = random_directions(N_q, dim) * norms_q.reshape(-1, 1)
    
    np.random.shuffle(db)
    np.random.shuffle(queries)
    
    return db, queries

if __name__ == "__main__":
    # Example usage
    db, queries = generate_two_densities(seed=42)
    print(f"Generated database: {db.shape}, queries: {queries.shape}")
    print(f"Norm range: [{np.linalg.norm(db, axis=1).min():.2f}, {np.linalg.norm(db, axis=1).max():.2f}]")