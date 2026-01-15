#!/usr/bin/env python3
"""
Cluster persons based on x0.25 OSNet similarity matrix.

Takes the comparison.json output and performs hierarchical clustering
to identify duplicate persons (same physical person at different times).
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

def load_similarity_matrix(json_file: str, model_key: str = 'model1'):
    """Load similarity matrix from comparison.json"""
    print(f"Loading {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if model_key not in data:
        raise ValueError(f"Model key '{model_key}' not found in JSON. Available: {list(data.keys())}")
    
    model_data = data[model_key]
    sim_matrix = np.array(model_data['similarity_matrix'])
    person_ids = model_data['person_ids']
    
    print(f"Loaded similarity matrix: {sim_matrix.shape}")
    print(f"Persons: {person_ids}")
    print(f"Mean similarity: {model_data['mean_similarity']:.4f}")
    print(f"Std similarity: {model_data['std_similarity']:.4f}")
    
    return sim_matrix, person_ids, model_data


def perform_clustering(sim_matrix: np.ndarray, person_ids: list, 
                       threshold: float = 0.7, method: str = 'ward') -> dict:
    """
    Perform hierarchical clustering on similarity matrix.
    
    Args:
        sim_matrix: (N, N) similarity matrix (higher = more similar)
        person_ids: List of person IDs
        threshold: Clustering threshold (persons more similar than this are clustered)
        method: Linkage method ('ward', 'complete', 'average', 'single')
    
    Returns:
        Dictionary with clustering results
    """
    print(f"\n{'='*70}")
    print(f"Clustering with threshold: {threshold}")
    print(f"Linkage method: {method}")
    
    # Convert similarity to distance: distance = 1 - similarity
    # Cap at 0 to handle numerical artifacts
    distance_matrix = np.maximum(1 - sim_matrix, 0)
    
    # Convert to condensed form for scipy
    condensed_dist = squareform(distance_matrix)
    
    # Perform hierarchical clustering
    Z = linkage(condensed_dist, method=method)
    
    # Get cluster assignments using threshold
    # Convert threshold from similarity to distance
    distance_threshold = 1 - threshold
    clusters = fcluster(Z, distance_threshold, criterion='distance')
    
    # Organize results
    n_clusters = len(np.unique(clusters))
    print(f"\nClustering Results:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Cluster assignments: {clusters}")
    
    # Print cluster details
    results = {
        'n_clusters': n_clusters,
        'clusters': {},
        'similarity_threshold': threshold,
        'distance_threshold': distance_threshold
    }
    
    for cluster_id in range(1, n_clusters + 1):
        members = [person_ids[i] for i in range(len(person_ids)) if clusters[i] == cluster_id]
        results['clusters'][f'cluster_{cluster_id}'] = {
            'members': members,
            'count': len(members)
        }
        print(f"\n  Cluster {cluster_id}: {len(members)} persons")
        for member in members:
            print(f"    - {member}")
    
    return results, Z, clusters


def analyze_within_cluster_similarity(sim_matrix: np.ndarray, person_ids: list, 
                                     clusters: np.ndarray) -> dict:
    """Analyze similarity statistics within clusters"""
    print(f"\n{'='*70}")
    print("Within-Cluster Similarity Analysis:")
    
    analysis = {}
    n_clusters = len(np.unique(clusters))
    
    for cluster_id in range(1, n_clusters + 1):
        member_indices = [i for i in range(len(person_ids)) if clusters[i] == cluster_id]
        
        if len(member_indices) < 2:
            continue
        
        # Extract similarities within cluster
        within_sims = []
        for i in range(len(member_indices)):
            for j in range(i+1, len(member_indices)):
                idx_i = member_indices[i]
                idx_j = member_indices[j]
                within_sims.append(sim_matrix[idx_i, idx_j])
        
        if within_sims:
            analysis[f'cluster_{cluster_id}'] = {
                'mean': float(np.mean(within_sims)),
                'min': float(np.min(within_sims)),
                'max': float(np.max(within_sims)),
                'std': float(np.std(within_sims))
            }
            print(f"\n  Cluster {cluster_id}:")
            print(f"    Mean within-cluster similarity: {np.mean(within_sims):.4f}")
            print(f"    Range: {np.min(within_sims):.4f} - {np.max(within_sims):.4f}")
            print(f"    Std: {np.std(within_sims):.4f}")
    
    return analysis


def plot_dendrogram(Z, person_ids: list, output_path: str):
    """Plot hierarchical clustering dendrogram"""
    print(f"\nGenerating dendrogram...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, labels=person_ids, ax=ax)
    ax.set_xlabel('Person')
    ax.set_ylabel('Distance (1 - similarity)')
    ax.set_title('Hierarchical Clustering of Persons (x0.25 OSNet)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    print(f"Dendrogram saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Cluster OSNet results')
    parser.add_argument('--json', type=str, required=True, help='Path to comparison.json')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.7, 
                       help='Similarity threshold for clustering (0-1, higher=more similar persons in same cluster)')
    parser.add_argument('--method', type=str, default='ward', 
                       choices=['ward', 'complete', 'average', 'single'],
                       help='Linkage method')
    parser.add_argument('--model', type=str, default='model1', 
                       help='Which model to cluster (model1 or model2)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    sim_matrix, person_ids, model_data = load_similarity_matrix(args.json, args.model)
    
    print(f"\nUsing model: {args.model}")
    print(f"  Type: {model_data['type']}")
    print(f"  Path: {model_data['path']}")
    
    # Perform clustering
    results, Z, clusters = perform_clustering(sim_matrix, person_ids, 
                                             threshold=args.threshold, 
                                             method=args.method)
    
    # Analyze within-cluster similarity
    similarity_analysis = analyze_within_cluster_similarity(sim_matrix, person_ids, clusters)
    
    # Plot dendrogram
    dendrogram_path = output_dir / 'dendrogram.png'
    plot_dendrogram(Z, person_ids, str(dendrogram_path))
    
    # Save results
    results['similarity_analysis'] = similarity_analysis
    results_file = output_dir / 'clustering_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to {results_file}")
    print(f"Dendrogram saved to {dendrogram_path}")


if __name__ == '__main__':
    main()
