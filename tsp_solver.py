#!/usr/bin/env python3
"""
TSP Solver Command Line Tool with PyConcorde
Load TSP instances and compute optimal solutions using PyConcorde (Concorde wrapper)
"""

import torch
import numpy as np
import argparse
import os
import time
import json
from typing import List, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Try to import PyConcorde (the correct Concorde package)
try:
    from concorde.tsp import TSPSolver
    PYCONCORDE_AVAILABLE = True
    print("✓ PyConcorde available")
except ImportError:
    PYCONCORDE_AVAILABLE = False
    print("✗ PyConcorde not available.")
    print("Install with: pip install -e git+https://github.com/jvkersch/pyconcorde.git#egg=pyconcorde")

# Try to import backup solvers
try:
    from python_tsp.exact import solve_tsp_dynamic_programming
    from python_tsp.heuristics import solve_tsp_simulated_annealing, solve_tsp_local_search
    PYTHON_TSP_AVAILABLE = True
    print("✓ python-tsp available")
except ImportError:
    PYTHON_TSP_AVAILABLE = False
    print("✗ python-tsp not available. Install with: pip install python-tsp")

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
    print("✓ OR-Tools available")
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("✗ OR-Tools not available. Install with: pip install ortools")

def load_tsp_dataset(filepath: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load TSP dataset from file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    data = torch.load(filepath, map_location='cpu')
    coordinates = data['coordinates']
    
    print(f"Dataset loaded from: {filepath}")
    print(f"Shape: {coordinates.shape}")
    print(f"Instances: {data['num_instances']}, Cities: {data['num_cities']}")
    
    return coordinates, data

def calculate_tour_length(tour: List[int], coords: torch.Tensor) -> float:
    """Calculate total tour length"""
    total_length = 0.0
    n_cities = len(tour)
    coords_np = coords.numpy()
    
    for i in range(n_cities):
        from_city = tour[i]
        to_city = tour[(i + 1) % n_cities]
        distance = np.sqrt(np.sum((coords_np[from_city] - coords_np[to_city]) ** 2))
        total_length += distance
    
    return total_length

def solve_with_pyconcorde(coords: torch.Tensor) -> Tuple[List[int], float, Dict[str, Any]]:
    """Solve TSP using PyConcorde (optimal solver)"""
    if not PYCONCORDE_AVAILABLE:
        raise ImportError("PyConcorde not available. Install with: pip install -e git+https://github.com/jvkersch/pyconcorde.git#egg=pyconcorde")
    
    coords_np = coords.numpy()
    
    print("  Using PyConcorde TSP solver (exact/optimal)...")
    start_time = time.time()
    
    # Create TSP solver from coordinates
    solver = TSPSolver.from_data(
        coords_np[:, 0],  # x coordinates
        coords_np[:, 1],  # y coordinates
        norm="EUC_2D"     # Euclidean 2D distance
    )
    
    # Solve
    solution = solver.solve()
    solve_time = time.time() - start_time
    
    tour = solution.tour.tolist()
    tour_length = calculate_tour_length(tour, coords)
    
    metadata = {
        'solver': 'pyconcorde',
        'optimal': True,
        'solve_time': solve_time,
        'found_tour': solution.found_tour,
        'optimal_value': solution.optimal_value,
        'concorde_optimal_value': solution.optimal_value
    }
    
    print(f"  ✓ Optimal tour found in {solve_time:.2f}s")
    print(f"  ✓ Concorde optimal value: {solution.optimal_value:.4f}")
    print(f"  ✓ Calculated tour length: {tour_length:.4f}")
    
    return tour, tour_length, metadata

def solve_with_python_tsp_exact(coords: torch.Tensor) -> Tuple[List[int], float, Dict[str, Any]]:
    """Solve TSP using python-tsp dynamic programming (exact for small instances)"""
    if not PYTHON_TSP_AVAILABLE:
        raise ImportError("python-tsp not available. Install with: pip install python-tsp")
    
    n_cities = len(coords)
    if n_cities > 20:
        raise ValueError(f"Dynamic programming solver too slow for {n_cities} cities (recommended ≤20)")
    
    coords_np = coords.numpy()
    
    # Calculate distance matrix
    distance_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                distance_matrix[i][j] = np.sqrt(np.sum((coords_np[i] - coords_np[j]) ** 2))
    
    print("  Using python-tsp dynamic programming (exact)...")
    start_time = time.time()
    
    # Solve using dynamic programming (exact but exponential time)
    tour, distance = solve_tsp_dynamic_programming(distance_matrix)
    solve_time = time.time() - start_time
    
    tour_length = calculate_tour_length(tour, coords)
    
    metadata = {
        'solver': 'python_tsp_dp',
        'optimal': True,
        'solve_time': solve_time,
        'reported_distance': distance
    }
    
    print(f"  ✓ Optimal tour found in {solve_time:.2f}s")
    return tour, tour_length, metadata

def solve_with_python_tsp_heuristic(coords: torch.Tensor) -> Tuple[List[int], float, Dict[str, Any]]:
    """Solve TSP using python-tsp simulated annealing (heuristic)"""
    if not PYTHON_TSP_AVAILABLE:
        raise ImportError("python-tsp not available. Install with: pip install python-tsp")
    
    coords_np = coords.numpy()
    n_cities = len(coords)
    
    # Calculate distance matrix
    distance_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                distance_matrix[i][j] = np.sqrt(np.sum((coords_np[i] - coords_np[j]) ** 2))
    
    print("  Using python-tsp simulated annealing (heuristic)...")
    start_time = time.time()
    
    # Solve using simulated annealing
    tour, distance = solve_tsp_simulated_annealing(distance_matrix)
    
    # Improve with local search
    tour, distance = solve_tsp_local_search(distance_matrix, tour)
    
    solve_time = time.time() - start_time
    tour_length = calculate_tour_length(tour, coords)
    
    metadata = {
        'solver': 'python_tsp_sa_ls',
        'optimal': False,
        'solve_time': solve_time,
        'reported_distance': distance
    }
    
    print(f"  ✓ Heuristic tour found in {solve_time:.2f}s")
    return tour, tour_length, metadata

def solve_with_ortools(coords: torch.Tensor, time_limit: int = 30) -> Tuple[List[int], float, Dict[str, Any]]:
    """Solve TSP using Google OR-Tools"""
    if not ORTOOLS_AVAILABLE:
        raise ImportError("OR-Tools not available. Install with: pip install ortools")
    
    coords_np = coords.numpy()
    n_cities = len(coords)
    
    # Calculate distance matrix
    distance_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                distance_matrix[i][j] = np.sqrt(np.sum((coords_np[i] - coords_np[j]) ** 2))
    
    print(f"  Using OR-Tools (time limit: {time_limit}s)...")
    start_time = time.time()
    
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(n_cities, 1, 0)
    
    # Create routing model
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 1000)  # Scale for integer
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = time_limit
    
    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    solve_time = time.time() - start_time
    
    if solution:
        # Extract tour
        tour = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            tour.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        
        tour_length = calculate_tour_length(tour, coords)
        
        metadata = {
            'solver': 'ortools',
            'optimal': False,  # OR-Tools doesn't guarantee optimality within time limit
            'solve_time': solve_time,
            'objective_value': solution.ObjectiveValue() / 1000.0,  # Unscale
            'status': 'solved'
        }
        
        print(f"  ✓ Tour found in {solve_time:.2f}s")
        return tour, tour_length, metadata
    else:
        raise RuntimeError("OR-Tools failed to find a solution")

def solve_single_instance(instance_idx: int, coords: torch.Tensor, solver: str, **kwargs) -> Dict[str, Any]:
    """Solve a single TSP instance"""
    n_cities = len(coords)
    
    print(f"\nSolving instance {instance_idx} ({n_cities} cities) with {solver}...")
    
    try:
        if solver == 'pyconcorde':
            tour, tour_length, metadata = solve_with_pyconcorde(coords)
        elif solver == 'python_tsp_exact':
            tour, tour_length, metadata = solve_with_python_tsp_exact(coords)
        elif solver == 'python_tsp_heuristic':
            tour, tour_length, metadata = solve_with_python_tsp_heuristic(coords)
        elif solver == 'ortools':
            time_limit = kwargs.get('time_limit', 30)
            tour, tour_length, metadata = solve_with_ortools(coords, time_limit)
        else:
            raise ValueError(f"Unknown solver: {solver}")
        
        result = {
            'instance_idx': instance_idx,
            'n_cities': n_cities,
            'tour': tour,
            'tour_length': tour_length,
            'coordinates': coords.tolist(),
            'success': True,
            'error': None,
            **metadata
        }
        
        print(f"  Instance {instance_idx}: Tour length = {tour_length:.4f}")
        return result
        
    except Exception as e:
        print(f"  ✗ Error solving instance {instance_idx}: {str(e)}")
        return {
            'instance_idx': instance_idx,
            'n_cities': n_cities,
            'tour': None,
            'tour_length': None,
            'coordinates': coords.tolist(),
            'success': False,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Solve TSP instances with PyConcorde and save optimal solutions')
    parser.add_argument('dataset_path', help='Path to TSP dataset (.pt file)')
    parser.add_argument('--solver', choices=['pyconcorde', 'python_tsp_exact', 'python_tsp_heuristic', 'ortools'], 
                       default='pyconcorde', help='TSP solver to use')
    parser.add_argument('--output', '-o', help='Output file path for results (default: auto-generated)')
    parser.add_argument('--max_instances', type=int, help='Maximum number of instances to solve')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--time_limit', type=int, default=30, help='Time limit for OR-Tools solver (seconds)')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting instance index')
    parser.add_argument('--end_idx', type=int, help='Ending instance index (exclusive)')
    
    args = parser.parse_args()
    
    # Check solver availability and provide fallbacks
    if args.solver == 'pyconcorde' and not PYCONCORDE_AVAILABLE:
        print("Error: PyConcorde not available.")
        print("Install with: pip install -e git+https://github.com/jvkersch/pyconcorde.git#egg=pyconcorde")
        if PYTHON_TSP_AVAILABLE:
            print("Falling back to python_tsp_heuristic solver...")
            args.solver = 'python_tsp_heuristic'
        elif ORTOOLS_AVAILABLE:
            print("Falling back to ortools solver...")
            args.solver = 'ortools'
        else:
            print("No suitable TSP solver available. Please install one of:")
            print("  - PyConcorde: pip install -e git+https://github.com/jvkersch/pyconcorde.git#egg=pyconcorde")
            print("  - python-tsp: pip install python-tsp")
            print("  - OR-Tools: pip install ortools")
            return
    elif args.solver.startswith('python_tsp') and not PYTHON_TSP_AVAILABLE:
        print("Error: python-tsp not available. Install with: pip install python-tsp")
        if PYCONCORDE_AVAILABLE:
            print("Falling back to pyconcorde solver...")
            args.solver = 'pyconcorde'
        elif ORTOOLS_AVAILABLE:
            print("Falling back to ortools solver...")
            args.solver = 'ortools'
        else:
            return
    elif args.solver == 'ortools' and not ORTOOLS_AVAILABLE:
        print("Error: OR-Tools not available. Install with: pip install ortools")
        if PYCONCORDE_AVAILABLE:
            print("Falling back to pyconcorde solver...")
            args.solver = 'pyconcorde'
        elif PYTHON_TSP_AVAILABLE:
            print("Falling back to python_tsp_heuristic solver...")
            args.solver = 'python_tsp_heuristic'
        else:
            return
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_path}")
    coordinates, metadata = load_tsp_dataset(args.dataset_path)
    
    # Determine instance range
    total_instances = len(coordinates)
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else total_instances
    
    if args.max_instances:
        end_idx = min(start_idx + args.max_instances, end_idx)
    
    print(f"Solving instances {start_idx} to {end_idx-1} ({end_idx-start_idx} total)")
    
    # Check if exact solver is suitable for problem size
    n_cities = coordinates.shape[1]
    if args.solver == 'python_tsp_exact' and n_cities > 20:
        print(f"Warning: Exact solver may be too slow for {n_cities} cities.")
        print("Consider using 'pyconcorde' or 'python_tsp_heuristic' for larger instances.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Solve instances
    results = []
    total_time = time.time()
    
    if args.parallel and (end_idx - start_idx) > 1:
        print(f"Using parallel processing with {mp.cpu_count()} workers...")
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = []
            for i in range(start_idx, end_idx):
                future = executor.submit(solve_single_instance, i, coordinates[i], 
                                       args.solver, time_limit=args.time_limit)
                futures.append(future)
            
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for i in range(start_idx, end_idx):
            result = solve_single_instance(i, coordinates[i], args.solver, 
                                         time_limit=args.time_limit)
            results.append(result)
    
    total_time = time.time() - total_time
    
    # Sort results by instance index
    results.sort(key=lambda x: x['instance_idx'])
    
    # Calculate statistics
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    if successful_results:
        tour_lengths = [r['tour_length'] for r in successful_results]
        solve_times = [r['solve_time'] for r in successful_results]
        
        print(f"\n=== RESULTS SUMMARY ===")
        print(f"Solver: {args.solver}")
        print(f"Total instances: {len(results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average solve time: {np.mean(solve_times):.2f}s")
        print(f"Tour length statistics:")
        print(f"  Min: {np.min(tour_lengths):.4f}")
        print(f"  Mean: {np.mean(tour_lengths):.4f}")
        print(f"  Max: {np.max(tour_lengths):.4f}")
        print(f"  Std: {np.std(tour_lengths):.4f}")
        
        if args.solver == 'pyconcorde':
            print(f"✓ All solutions are OPTIMAL (guaranteed by Concorde)")
    
    # Generate output filename if not provided
    if args.output is None:
        dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
        args.output = f"{dataset_name}_{args.solver}_solutions.json"
    
    # Save results
    output_data = {
        'dataset_path': args.dataset_path,
        'dataset_metadata': metadata,
        'solver': args.solver,
        'solver_params': {'time_limit': args.time_limit} if args.solver == 'ortools' else {},
        'total_instances': len(results),
        'successful_instances': len(successful_results),
        'failed_instances': len(failed_results),
        'total_solve_time': total_time,
        'optimal_guaranteed': args.solver in ['pyconcorde', 'python_tsp_exact'],
        'results': results
    }
    
    print(f"\nSaving results to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved successfully!")
    
    if failed_results:
        print(f"\nFailed instances:")
        for result in failed_results:
            print(f"  Instance {result['instance_idx']}: {result['error']}")

if __name__ == "__main__":
    main()
