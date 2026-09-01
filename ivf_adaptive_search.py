"""
Main Experimental Script for Norm-Based Adaptive Probe Selection
Implements IVF index with adaptive nprobe selection heuristic
"""

import numpy as np
import time
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
import csv
import os

# Import dataset generators
from data_generation import generate_two_densities, generate_three_densities, generate_pareto_tails


def get_ground_truth(db, queries, k=10):
    """Compute exact ground truth top-k neighbors"""
    dists = cdist(queries, db, metric='euclidean')
    return np.argsort(dists, axis=1)[:, :k]


def search_ivf_fixed(queries, db, centroids, labels, p_probe):
    """
    Standard IVF search with fixed nprobe
    Returns: (avg_distance_computations, retrieved_indices_list)
    """
    total_dist_comps = 0
    all_retrieved = []
    
    for q in queries:
        # Distance to all centroids
        centroid_dists = np.linalg.norm(centroids - q, axis=1)
        top_c_indices = np.argsort(centroid_dists)[:p_probe]
        
        # Collect vectors from selected clusters
        candidate_indices = []
        for c_idx in top_c_indices:
            candidate_indices.extend(np.where(labels == c_idx)[0])
        
        candidate_indices = np.unique(candidate_indices)
        total_dist_comps += len(candidate_indices)
        
        # Exact search within candidates
        candidate_dists = np.linalg.norm(db[candidate_indices] - q, axis=1)
        top_k_in_candidates = np.argsort(candidate_dists)[:10]
        retrieved_indices = candidate_indices[top_k_in_candidates]
        all_retrieved.append(retrieved_indices)
    
    return total_dist_comps / len(queries), all_retrieved


def search_ivf_adaptive(queries, db, centroids, labels, p0, alpha):
    """
    Adaptive IVF search with norm-based nprobe selection
    Returns: (avg_distance_computations, retrieved_indices_list, avg_nprobe)
    """
    total_dist_comps = 0
    all_retrieved = []
    total_nprobe = 0
    
    # Precompute centroid statistics
    centroid_norms = np.linalg.norm(centroids, axis=1)
    mu = np.mean(centroid_norms)
    sigma = np.std(centroid_norms)
    C = len(centroids)
    
    for q in queries:
        q_norm = np.linalg.norm(q)
        
        # Adaptive nprobe based on query norm Z-score
        z_score = (q_norm - mu) / sigma
        p = int(np.clip(p0 * (1 + alpha * z_score), 1, C))
        total_nprobe += p
        
        # Distance to all centroids
        centroid_dists = np.linalg.norm(centroids - q, axis=1)
        top_c_indices = np.argsort(centroid_dists)[:p]
        
        # Collect vectors from selected clusters
        candidate_indices = []
        for c_idx in top_c_indices:
            candidate_indices.extend(np.where(labels == c_idx)[0])
        
        candidate_indices = np.unique(candidate_indices)
        total_dist_comps += len(candidate_indices)
        
        # Exact search within candidates
        candidate_dists = np.linalg.norm(db[candidate_indices] - q, axis=1)
        top_k_in_candidates = np.argsort(candidate_dists)[:10]
        retrieved_indices = candidate_indices[top_k_in_candidates]
        all_retrieved.append(retrieved_indices)
    
    return (total_dist_comps / len(queries), 
            all_retrieved, 
            total_nprobe / len(queries))


def compute_recall(retrieved_list, ground_truth):
    """Compute recall@10"""
    recalls = []
    for ret, gt in zip(retrieved_list, ground_truth):
        intersection = len(set(ret).intersection(set(gt)))
        recalls.append(intersection / 10)
    return np.mean(recalls)


def run_experiment(seed, config='two'):
    """Run full experiment for one seed and configuration"""
    print(f"Running seed {seed} ({config})...")
    
    # Generate dataset
    if config == 'two':
        db, queries = generate_two_densities(seed=seed)
    elif config == 'three':
        db, queries = generate_three_densities(seed=seed)
    else:
        db, queries = generate_pareto_tails(seed=seed)
    
    # Build IVF index
    kmeans = MiniBatchKMeans(n_clusters=100, random_state=seed, batch_size=1024)
    labels = kmeans.fit_predict(db)
    centroids = kmeans.cluster_centers_
    
    # Ground truth
    gt = get_ground_truth(db, queries, k=10)
    
    # Sweep over p0 values
    p0_values = [2, 5, 10, 15, 25, 40, 70]
    alpha = 0.3
    
    results = []
    
    for p0 in p0_values:
        # Fixed baseline
        dist_fixed, ret_fixed = search_ivf_fixed(queries, db, centroids, labels, p0)
        recall_fixed = compute_recall(ret_fixed, gt)
        
        # Adaptive
        dist_adapt, ret_adapt, avg_nprobe = search_ivf_adaptive(
            queries, db, centroids, labels, p0, alpha)
        recall_adapt = compute_recall(ret_adapt, gt)
        
        results.append({
            'p0': p0,
            'recall_fixed': recall_fixed,
            'dist_fixed': dist_fixed,
            'recall_adapt': recall_adapt,
            'dist_adapt': dist_adapt,
            'avg_nprobe': avg_nprobe
        })
    
    return results


def save_results(all_results, filename):
    """Save results to CSV"""
    os.makedirs('data', exist_ok=True)
    
    with open(f'data/{filename}', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'seed', 'config', 'p0', 'recall_fixed', 'dist_fixed',
            'recall_adapt', 'dist_adapt', 'avg_nprobe'
        ])
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"Results saved to data/{filename}")


if __name__ == "__main__":
    # Run experiments across multiple seeds
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    configs = ['two', 'three']
    
    all_results = []
    
    for config in configs:
        for seed in seeds:
            results = run_experiment(seed, config)
            for r in results:
                r['seed'] = seed
                r['config'] = config
                all_results.append(r)
    
    save_results(all_results, 'experimental_results.csv')
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)