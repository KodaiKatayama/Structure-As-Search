import torch
import numpy as np
import os

# VRPの車両容量（都市数に応じた容量設定）
VRP_CAPACITIES = {
    5: 15,
    20: 30,
    50: 40,
    100: 50
}

# VRPデータ生成関数
def generate_vrp_data(num_instances, num_cities, coord_range=100.0, demand_range=(1,10), seed=None):
    """
    ランダムなVRPインスタンスを生成
    
    引数:
        num_instances: VRPインスタンスの数
        num_cities: インスタンスあたりの都市数
        coord_range: 座標範囲 [0, coord_range]
        demand_range: 需要の範囲を示すタプル (最小, 最大)
        seed: 再現性のためのランダムシード
    
    戻り値:
        depot_coordinates: (num_instances, 2) tensor
        customer_coordinates: (num_instances, num_cities, 2) tensor
        raw_demands: (num_instances, num_cities) tensor
        normalized_demands: (num_instances, num_cities) tensor
        capacity: Vehicle capacity (scalar)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    capacity = VRP_CAPACITIES.get(num_cities, 50)  # Default capacity if num_cities not in dict
    
    # Generate random depot locations
    depot_coordinates = torch.rand(num_instances, 2) * coord_range

    # Generate random coordinates for cities
    customer_coordinates = torch.rand(num_instances, num_cities, 2) * coord_range

    # Generate random demands for each city
    raw_demands = torch.randint(demand_range[0], demand_range[1], (num_instances, num_cities)).float()
    normalized_demands = raw_demands / capacity  # Normalize demands by capacity
    
    return {
        'depot_coordinates': depot_coordinates,
        'customer_coordinates': customer_coordinates,
        'raw_demands': raw_demands,
        'normalized_demands': normalized_demands,
        'capacity': capacity
    }