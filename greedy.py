"""
Greedy TSP Solver
Load validation data and compute TSP tour lengths using greedy nearest neighbor algorithm
"""

import torch
import numpy as np
from tqdm import tqdm
import time
import argparse

def load_tsp_dataset(filepath):
    """Load TSP dataset"""
    data = torch.load(filepath)
    coordinates = data['coordinates']
    print(f"Dataset loaded: {coordinates.shape}")
    return coordinates, data

def compute_distance_matrix(coordinates):
    """
    Compute distance matrix from coordinates
    
    Args:
        coordinates: (N, 2) coordinates of cities
    
    Returns:
        distance_matrix: (N, N) pairwise distances
    """
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.cpu().numpy()
    
    n = len(coordinates)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coordinates[i, 0] - coordinates[j, 0]
                dy = coordinates[i, 1] - coordinates[j, 1]
                distance_matrix[i, j] = np.sqrt(dx*dx + dy*dy)
    
    return distance_matrix

def greedy_tsp(coordinates, start_city=0):
    """
    Solve TSP using greedy nearest neighbor algorithm
    
    Args:
        coordinates: (N, 2) numpy array of city coordinates
        start_city: starting city index
    
    Returns:
        tour: list of city indices in tour order
        tour_length: total tour length
    """
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.cpu().numpy()
    
    n = len(coordinates)
    
    # Compute distance matrix
    distance_matrix = compute_distance_matrix(coordinates)
    
    # Initialize
    tour = [start_city]
    unvisited = set(range(n))
    unvisited.remove(start_city)
    current_city = start_city
    tour_length = 0.0
    
    # Greedy construction
    while unvisited:
        # Find nearest unvisited city
        nearest_city = None
        nearest_distance = float('inf')
        
        for city in unvisited:
            distance = distance_matrix[current_city, city]
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_city = city
        
        # Move to nearest city
        tour.append(nearest_city)
        tour_length += nearest_distance
        unvisited.remove(nearest_city)
        current_city = nearest_city
    
    # Return to start city
    tour.append(start_city)
    tour_length += distance_matrix[current_city, start_city]
    
    return tour, tour_length

def greedy_tsp_optimized(coordinates, start_city=0):
    """
    Optimized greedy TSP using vectorized operations
    
    Args:
        coordinates: (N, 2) numpy array of city coordinates
        start_city: starting city index
    
    Returns:
        tour: list of city indices in tour order
        tour_length: total tour length
    """
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.cpu().numpy()
    
    n = len(coordinates)
    
    # Compute distance matrix using broadcasting
    coords_expanded = coordinates[:, np.newaxis, :]  # (N, 1, 2)
    distance_matrix = np.sqrt(np.sum((coords_expanded - coordinates)**2, axis=2))  # (N, N)
    
    # Initialize
    tour = [start_city]
    visited = np.zeros(n, dtype=bool)
    visited[start_city] = True
    current_city = start_city
    tour_length = 0.0
    
    # Greedy construction
    for _ in range(n - 1):
        # Get distances from current city to all unvisited cities
        distances = distance_matrix[current_city].copy()
        distances[visited] = np.inf  # Mask visited cities
        
        # Find nearest unvisited city
        nearest_city = np.argmin(distances)
        nearest_distance = distances[nearest_city]
        
        # Move to nearest city
        tour.append(nearest_city)
        tour_length += nearest_distance
        visited[nearest_city] = True
        current_city = nearest_city
    
    # Return to start city
    tour.append(start_city)
    tour_length += distance_matrix[current_city, start_city]
    
    return tour, tour_length

def greedy_tsp_multi_start(coordinates, num_starts=None):
    """
    Run greedy TSP from multiple starting cities and return best result
    
    Args:
        coordinates: (N, 2) coordinates of cities
        num_starts: number of different starting cities to try (default: all cities)
    
    Returns:
        best_tour: best tour found
        best_length: length of best tour
        all_results: list of (tour, length) for all starts
    """
    n = len(coordinates)
    if num_starts is None:
        num_starts = n
    
    best_tour = None
    best_length = float('inf')
    all_results = []
    
    # Try different starting cities
    start_cities = range(min(num_starts, n))
    
    for start_city in start_cities:
        tour, length = greedy_tsp_optimized(coordinates, start_city)
        all_results.append((tour, length))
        
        if length < best_length:
            best_length = length
            best_tour = tour
    
    return best_tour, best_length, all_results

def solve_tsp_dataset(coordinates_batch, method='single_start', num_starts=1, show_progress=True):
    """
    Solve TSP for entire dataset
    
    Args:
        coordinates_batch: (B, N, 2) batch of TSP instances
        method: 'single_start' or 'multi_start'
        num_starts: number of starting cities for multi-start
        show_progress: whether to show progress bar
    
    Returns:
        tour_lengths: list of tour lengths
        tours: list of tours
        stats: dictionary with statistics
    """
    if isinstance(coordinates_batch, torch.Tensor):
        coordinates_batch = coordinates_batch.cpu().numpy()
    
    batch_size = len(coordinates_batch)
    tour_lengths = []
    tours = []
    
    iterator = tqdm(range(batch_size), desc="Solving TSP") if show_progress else range(batch_size)
    
    start_time = time.time()
    
    for i in iterator:
        coordinates = coordinates_batch[i]
        
        if method == 'single_start':
            tour, length = greedy_tsp_optimized(coordinates, start_city=0)
        elif method == 'multi_start':
            tour, length, _ = greedy_tsp_multi_start(coordinates, num_starts=num_starts)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        tour_lengths.append(length)
        tours.append(tour)
    
    end_time = time.time()
    
    # Compute statistics
    tour_lengths = np.array(tour_lengths)
    stats = {
        'mean_length': np.mean(tour_lengths),
        'std_length': np.std(tour_lengths),
        'min_length': np.min(tour_lengths),
        'max_length': np.max(tour_lengths),
        'median_length': np.median(tour_lengths),
        'total_time': end_time - start_time,
        'avg_time_per_instance': (end_time - start_time) / batch_size,
        'method': method,
        'num_starts': num_starts if method == 'multi_start' else 1
    }
    
    return tour_lengths.tolist(), tours, stats

def compare_methods(coordinates_batch, max_instances=100):
    """
    Compare different greedy methods
    
    Args:
        coordinates_batch: (B, N, 2) batch of TSP instances
        max_instances: maximum number of instances to test
    
    Returns:
        comparison_results: dictionary with results for each method
    """
    if len(coordinates_batch) > max_instances:
        coordinates_batch = coordinates_batch[:max_instances]
        print(f"Using first {max_instances} instances for comparison")
    
    methods = [
        ('Single Start (0)', 'single_start', 1),
        ('Multi Start (5)', 'multi_start', 5),
        ('Multi Start (All)', 'multi_start', None)
    ]
    
    results = {}
    
    for method_name, method_type, num_starts in methods:
        print(f"\n=== {method_name} ===")
        
        lengths, tours, stats = solve_tsp_dataset(
            coordinates_batch, 
            method=method_type, 
            num_starts=num_starts,
            show_progress=True
        )
        
        results[method_name] = {
            'lengths': lengths,
            'tours': tours,
            'stats': stats
        }
        
        print(f"Mean tour length: {stats['mean_length']:.2f}")
        print(f"Std deviation: {stats['std_length']:.2f}")
        print(f"Best tour: {stats['min_length']:.2f}")
        print(f"Worst tour: {stats['max_length']:.2f}")
        print(f"Total time: {stats['total_time']:.2f}s")
        print(f"Avg time per instance: {stats['avg_time_per_instance']:.4f}s")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Greedy TSP Solver')
    parser.add_argument('--data_path', type=str, default='data/tsp_50_uniform_test.pt',
                        help='Path to TSP dataset')
    parser.add_argument('--method', type=str, default='single_start',
                        choices=['single_start', 'multi_start'],
                        help='Greedy method to use')
    parser.add_argument('--num_starts', type=int, default=10,
                        help='Number of starting cities for multi-start')
    parser.add_argument('--max_instances', type=int, default=1000,
                        help='Maximum number of instances to solve')
    parser.add_argument('--compare', action='store_true',
                        help='Compare different methods')
    
    args = parser.parse_args()
    
    print("=== Greedy TSP Solver ===")
    print(f"Data path: {args.data_path}")
    
    # Load dataset
    coordinates_batch, metadata = load_tsp_dataset(args.data_path)
    
    print(f"Dataset info:")
    print(f"  Shape: {coordinates_batch.shape}")
    print(f"  Instances: {len(coordinates_batch)}")
    print(f"  Cities per instance: {coordinates_batch.shape[1]}")
    print(f"  Coordinate range: [{coordinates_batch.min():.1f}, {coordinates_batch.max():.1f}]")
    
    # Limit instances if requested
    if len(coordinates_batch) > args.max_instances:
        coordinates_batch = coordinates_batch[:args.max_instances]
        print(f"Using first {args.max_instances} instances")
    
    if args.compare:
        # Compare different methods
        print(f"\n=== Comparing Methods ===")
        results = compare_methods(coordinates_batch, max_instances=min(100, len(coordinates_batch)))
        
        # Summary comparison
        print(f"\n=== Method Comparison Summary ===")
        for method_name, result in results.items():
            stats = result['stats']
            print(f"{method_name:20}: Mean={stats['mean_length']:7.2f}, "
                  f"Best={stats['min_length']:7.2f}, Time={stats['avg_time_per_instance']:6.4f}s/inst")
    
    else:
        # Solve using specified method
        print(f"\n=== Solving with {args.method} method ===")
        if args.method == 'multi_start':
            print(f"Number of starting cities: {args.num_starts}")
        
        lengths, tours, stats = solve_tsp_dataset(
            coordinates_batch,
            method=args.method,
            num_starts=args.num_starts
        )
        
        # Print results
        print(f"\n=== Results ===")
        print(f"Instances solved: {len(lengths)}")
        print(f"Mean tour length: {stats['mean_length']:.2f}")
        print(f"Standard deviation: {stats['std_length']:.2f}")
        print(f"Best tour length: {stats['min_length']:.2f}")
        print(f"Worst tour length: {stats['max_length']:.2f}")
        print(f"Median tour length: {stats['median_length']:.2f}")
        print(f"Total solving time: {stats['total_time']:.2f}s")
        print(f"Average time per instance: {stats['avg_time_per_instance']:.4f}s")
        
        # Show some individual results
        print(f"\nFirst 10 tour lengths:")
        for i, length in enumerate(lengths[:10]):
            print(f"  Instance {i}: {length:.2f}")
        
        # Save results
        result_file = f"greedy_results_{args.method}.txt"
        with open(result_file, 'w') as f:
            f.write(f"Greedy TSP Results - {args.method}\n")
            f.write("=" * 40 + "\n")
            f.write(f"Dataset: {args.data_path}\n")
            f.write(f"Method: {args.method}\n")
            if args.method == 'multi_start':
                f.write(f"Number of starts: {args.num_starts}\n")
            f.write(f"Instances: {len(lengths)}\n")
            f.write(f"Mean length: {stats['mean_length']:.2f}\n")
            f.write(f"Std deviation: {stats['std_length']:.2f}\n")
            f.write(f"Best length: {stats['min_length']:.2f}\n")
            f.write(f"Worst length: {stats['max_length']:.2f}\n")
            f.write(f"Median length: {stats['median_length']:.2f}\n")
            f.write(f"Total time: {stats['total_time']:.2f}s\n")
            f.write(f"Avg time per instance: {stats['avg_time_per_instance']:.4f}s\n")
            f.write(f"\nAll tour lengths:\n")
            for i, length in enumerate(lengths):
                f.write(f"{i:4d}: {length:.2f}\n")
        
        print(f"\nResults saved to: {result_file}")

def test_greedy():
    """Test the greedy solver with small example"""
    print("Testing greedy TSP solver...")
    
    # Create simple test case
    coordinates = np.array([
        [0, 0],
        [1, 0], 
        [1, 1],
        [0, 1]
    ])
    
    print(f"Test coordinates:\n{coordinates}")
    
    # Test single start
    tour, length = greedy_tsp_optimized(coordinates, start_city=0)
    print(f"Greedy tour from city 0: {tour}")
    print(f"Tour length: {length:.2f}")
    
    # Test multi start
    best_tour, best_length, all_results = greedy_tsp_multi_start(coordinates)
    print(f"Best tour (multi-start): {best_tour}")
    print(f"Best length: {best_length:.2f}")
    
    print("✓ Greedy solver test completed!")

if __name__ == "__main__":
    # Uncomment to run test
    # test_greedy()
    
    # Run main solver
    main()
