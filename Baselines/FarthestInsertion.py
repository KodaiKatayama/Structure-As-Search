"""
TSP Farthest Insertion Algorithm Implementation
Load test data and solve using farthest insertion heuristic
"""

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from typing import List, Tuple
import time
import sys
import os

def load_tsp_dataset(filepath):
    """
    Load TSP dataset from file
    
    Args:
        filepath: Path to dataset file
    
    Returns:
        coordinates: (B, N, 2) coordinate tensor
        metadata: Dictionary with dataset info
    """
    import os
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    data = torch.load(filepath)
    coordinates = data['coordinates']
    
    print(f"Dataset loaded from: {filepath}")
    print(f"Shape: {coordinates.shape}")
    print(f"Instances: {data['num_instances']}, Cities: {data['num_cities']}")
    
    return coordinates, data

def calculate_distance_matrix(coordinates):
    """
    Calculate Euclidean distance matrix from coordinates
    
    Args:
        coordinates: (N, 2) coordinate tensor for single instance
    
    Returns:
        distance_matrix: (N, N) distance matrix
    """
    return torch.cdist(coordinates.unsqueeze(0), coordinates.unsqueeze(0)).squeeze(0)

def calculate_tour_length(tour: List[int], distance_matrix: torch.Tensor) -> float:
    """
    Calculate total length of a tour
    
    Args:
        tour: List of city indices in order
        distance_matrix: (N, N) distance matrix
    
    Returns:
        total_length: Total tour length
    """
    total_length = 0.0
    for i in range(len(tour)):
        current_city = tour[i]
        next_city = tour[(i + 1) % len(tour)]  # Wrap around to complete the cycle
        total_length += distance_matrix[current_city, next_city].item()
    
    return total_length

def find_farthest_city(partial_tour: List[int], distance_matrix: torch.Tensor, 
                      remaining_cities: List[int]) -> int:
    """
    Find the city that is farthest from the current partial tour
    
    Args:
        partial_tour: List of cities already in the tour
        distance_matrix: (N, N) distance matrix
        remaining_cities: List of cities not yet in the tour
    
    Returns:
        farthest_city: Index of the farthest city
    """
    max_min_distance = -1
    farthest_city = -1
    
    for city in remaining_cities:
        # Find minimum distance from this city to any city in the partial tour
        min_distance_to_tour = float('inf')
        for tour_city in partial_tour:
            distance = distance_matrix[city, tour_city].item()
            min_distance_to_tour = min(min_distance_to_tour, distance)
        
        # Keep track of the city with maximum minimum distance
        if min_distance_to_tour > max_min_distance:
            max_min_distance = min_distance_to_tour
            farthest_city = city
    
    return farthest_city

def find_best_insertion_position(tour: List[int], new_city: int, 
                                distance_matrix: torch.Tensor) -> int:
    """
    Find the best position to insert a new city in the tour
    
    Args:
        tour: Current tour
        new_city: City to insert
        distance_matrix: (N, N) distance matrix
    
    Returns:
        best_position: Best position to insert the new city
    """
    best_position = 0
    min_increase = float('inf')
    
    for i in range(len(tour)):
        # Calculate cost increase of inserting new_city between tour[i] and tour[(i+1)%len(tour)]
        current_city = tour[i]
        next_city = tour[(i + 1) % len(tour)]
        
        # Current edge cost
        current_cost = distance_matrix[current_city, next_city].item()
        
        # New cost with insertion
        new_cost = (distance_matrix[current_city, new_city].item() + 
                   distance_matrix[new_city, next_city].item())
        
        # Cost increase
        cost_increase = new_cost - current_cost
        
        if cost_increase < min_increase:
            min_increase = cost_increase
            best_position = i + 1
    
    return best_position

def farthest_insertion_tsp(coordinates: torch.Tensor, verbose: bool = False) -> Tuple[List[int], float]:
    """
    Solve TSP using Farthest Insertion algorithm
    
    Args:
        coordinates: (N, 2) coordinate tensor for single TSP instance
        verbose: Whether to print progress
    
    Returns:
        tour: List of city indices in tour order
        tour_length: Total length of the tour
    """
    n_cities = coordinates.shape[0]
    distance_matrix = calculate_distance_matrix(coordinates)
    
    # Step 1: Find the two cities that are farthest apart
    max_distance = -1
    start_city1, start_city2 = 0, 1
    
    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            distance = distance_matrix[i, j].item()
            if distance > max_distance:
                max_distance = distance
                start_city1, start_city2 = i, j
    
    # Initialize tour with the two farthest cities
    tour = [start_city1, start_city2]
    remaining_cities = [i for i in range(n_cities) if i not in tour]
    
    if verbose:
        print(f"Starting with cities {start_city1} and {start_city2} (distance: {max_distance:.2f})")
    
    # Step 2: Iteratively add the farthest city to the tour
    while remaining_cities:
        # Find the city that is farthest from the current tour
        farthest_city = find_farthest_city(tour, distance_matrix, remaining_cities)
        
        # Find the best position to insert this city
        best_position = find_best_insertion_position(tour, farthest_city, distance_matrix)
        
        # Insert the city at the best position
        tour.insert(best_position, farthest_city)
        remaining_cities.remove(farthest_city)
        
        if verbose:
            print(f"Added city {farthest_city} at position {best_position}, tour length: {len(tour)}")
    
    # Calculate final tour length
    tour_length = calculate_tour_length(tour, distance_matrix)
    
    return tour, tour_length

def visualize_tsp_solution(coordinates: torch.Tensor, tour: List[int], 
                          title: str = "TSP Solution - Farthest Insertion"):
    """
    Visualize the TSP solution
    
    Args:
        coordinates: (N, 2) coordinate tensor
        tour: List of city indices in tour order
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Convert to numpy for plotting
    coords = coordinates.numpy()
    
    # Plot cities
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=3)
    
    # Plot tour edges
    for i in range(len(tour)):
        current_city = tour[i]
        next_city = tour[(i + 1) % len(tour)]
        
        x_coords = [coords[current_city, 0], coords[next_city, 0]]
        y_coords = [coords[current_city, 1], coords[next_city, 1]]
        
        plt.plot(x_coords, y_coords, 'b-', alpha=0.7, linewidth=2)
    
    # Highlight start city
    start_city = tour[0]
    plt.scatter(coords[start_city, 0], coords[start_city, 1], 
               c='green', s=200, marker='*', zorder=4, label='Start')
    
    # Add city labels
    for i, (x, y) in enumerate(coords):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def test_farthest_insertion(dataset_path):
    """
    Test the farthest insertion algorithm on TSP data and report statistics
    
    Args:
        dataset_path: Path to the TSP dataset file
    """
    print("=== Testing Farthest Insertion TSP Algorithm ===\n")
    
    # Load the specified dataset
    try:
        test_coords, test_metadata = load_tsp_dataset(dataset_path)
        print(f"Loaded test dataset: {test_coords.shape}")
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please check the file path and try again.")
        return None, None
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return None, None
    
    # Solve all instances and collect results
    num_instances = test_coords.shape[0]
    tour_lengths = []
    solve_times = []
    
    print(f"\nSolving {num_instances} TSP instances with Farthest Insertion...")
    print("=" * 60)
    
    for instance_idx in range(num_instances):
        print(f"Solving instance {instance_idx + 1}/{num_instances}...", end=" ")
        
        coordinates = test_coords[instance_idx]  # (N, 2)
        n_cities = coordinates.shape[0]
        
        # Solve using farthest insertion
        start_time = time.time()
        tour, tour_length = farthest_insertion_tsp(coordinates)
        solve_time = time.time() - start_time
        
        tour_lengths.append(tour_length)
        solve_times.append(solve_time)
        
        print(f"Length: {tour_length:.2f}, Time: {solve_time:.3f}s")
        
        # Verify tour validity
        assert len(tour) == n_cities, "Tour should visit all cities"
        assert len(set(tour)) == n_cities, "Tour should visit each city exactly once"
    
    # Convert to numpy arrays for statistics
    tour_lengths = np.array(tour_lengths)
    solve_times = np.array(solve_times)
    
    # Calculate and report statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Number of instances: {num_instances}")
    print(f"Cities per instance: {test_coords.shape[1]}")
    print()
    print("TOUR LENGTH STATISTICS:")
    print(f"  Mean:     {tour_lengths.mean():.2f}")
    print(f"  Variance: {tour_lengths.var():.2f}")
    print(f"  Std Dev:  {tour_lengths.std():.2f}")
    print(f"  Min:      {tour_lengths.min():.2f}")
    print(f"  Max:      {tour_lengths.max():.2f}")
    print(f"  Median:   {np.median(tour_lengths):.2f}")
    print()
    print("SOLVING TIME STATISTICS:")
    print(f"  Mean time: {solve_times.mean():.3f}s")
    print(f"  Total time: {solve_times.sum():.2f}s")
    print()
    
    # Show distribution of tour lengths
    plt.figure(figsize=(12, 5))
    
    # Histogram of tour lengths
    plt.subplot(1, 2, 1)
    plt.hist(tour_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(tour_lengths.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {tour_lengths.mean():.2f}')
    plt.axvline(np.median(tour_lengths), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(tour_lengths):.2f}')
    plt.xlabel('Tour Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Tour Lengths\nFarthest Insertion Algorithm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(tour_lengths, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    plt.ylabel('Tour Length')
    plt.title('Tour Length Distribution\n(Box Plot)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize a few sample solutions
    print("\nVisualizing sample solutions...")
    sample_indices = [0, len(tour_lengths)//2, len(tour_lengths)-1]  # First, middle, last
    
    for i, idx in enumerate(sample_indices):
        if idx < num_instances:
            coordinates = test_coords[idx]
            tour, _ = farthest_insertion_tsp(coordinates)
            visualize_tsp_solution(coordinates, tour, 
                                 f"Sample {i+1}: Instance {idx+1} - Length: {tour_lengths[idx]:.2f}")
    
    return tour_lengths, solve_times

def compare_with_random_tour():
    """
    Compare farthest insertion with a random tour
    """
    print("\n=== Comparison with Random Tour ===")
    
    # Generate a small test instance
    torch.manual_seed(123)
    coordinates = torch.rand(15, 2) * 100.0
    
    # Solve with farthest insertion
    tour_fi, length_fi = farthest_insertion_tsp(coordinates)
    
    # Generate random tour
    random_tour = list(range(15))
    torch.manual_seed(456)
    random_indices = torch.randperm(15).tolist()
    random_tour = random_indices
    
    distance_matrix = calculate_distance_matrix(coordinates)
    length_random = calculate_tour_length(random_tour, distance_matrix)
    
    print(f"Farthest Insertion - Tour length: {length_fi:.2f}")
    print(f"Random Tour - Tour length: {length_random:.2f}")
    print(f"Improvement: {((length_random - length_fi) / length_random * 100):.1f}%")
    
    # Visualize both solutions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    coords = coordinates.numpy()
    
    # Plot farthest insertion solution
    ax1.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=3)
    for i in range(len(tour_fi)):
        current_city = tour_fi[i]
        next_city = tour_fi[(i + 1) % len(tour_fi)]
        x_coords = [coords[current_city, 0], coords[next_city, 0]]
        y_coords = [coords[current_city, 1], coords[next_city, 1]]
        ax1.plot(x_coords, y_coords, 'b-', alpha=0.7, linewidth=2)
    
    ax1.set_title(f'Farthest Insertion\nLength: {length_fi:.2f}')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot random solution
    ax2.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=3)
    for i in range(len(random_tour)):
        current_city = random_tour[i]
        next_city = random_tour[(i + 1) % len(random_tour)]
        x_coords = [coords[current_city, 0], coords[next_city, 0]]
        y_coords = [coords[current_city, 1], coords[next_city, 1]]
        ax2.plot(x_coords, y_coords, 'r-', alpha=0.7, linewidth=2)
    
    ax2.set_title(f'Random Tour\nLength: {length_random:.2f}')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Solve TSP using Farthest Insertion Algorithm')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the TSP dataset file (e.g., ../data/tsp_100_uniform_test.pt)')
    
    args = parser.parse_args()
    dataset_path = args.dataset_path
    
    # Check if file exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: File does not exist: {dataset_path}")
        print("Please check the file path and try again.")
        sys.exit(1)
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Test the farthest insertion algorithm
    tour_lengths, solve_times = test_farthest_insertion(dataset_path)
    
    if tour_lengths is not None and solve_times is not None:
        print(f"\n=== Final Summary ===")
        print(f"Processed {len(tour_lengths)} TSP instances")
        print(f"Mean tour length: {np.mean(tour_lengths):.2f}")
        print(f"Variance: {np.var(tour_lengths):.2f}")
        print(f"Standard deviation: {np.std(tour_lengths):.2f}")
        print(f"Mean solve time: {np.mean(solve_times):.3f}s")
        print("=== Farthest Insertion TSP Complete ===")
    else:
        print("Failed to process dataset.")
        sys.exit(1)
