"""
シンプルなVRPデータジェネレーター
座標(x, y)と需要を含む2つの特徴を持つデータを生成
SCTモデル向けの保存、ロード、データセット機能を備える
"""

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

# 座標を距離行列に変換
def coords_to_distance_matrix(coordinates):
    """Convert coordinates to distance matrix"""
    # coordinates: (B, N, 2) -> distance_matrix: (B, N, N)
    distance_matrix = torch.cdist(coordinates, coordinates)
    return distance_matrix

# データセットをファイルに保存する関数
def save_vrp_dataset(data_dict, filepath, metadata=None):
    """
    VRPデータセットをファイルに保存
    
    引数:
        data_dict: generate_vrp_dataが返した辞書
        filepath: 保存先のパス
        metadata: 追加のメタデータ辞書
    """
    # 保存先のディレクトリを作成
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # 保存用データの整理
    save_data = data_dict.copy()
    
    if metadata:
        save_data.update(metadata)
    
    # ファイルに保存
    torch.save(save_data, filepath)
    
    print(f"Dataset saved to: {filepath}")
    print(f"Instances: {data_dict['depot_coordinates'].shape[0]}, Cities: {data_dict['customer_coordinates'].shape[1]}")
    print(f"Capacity: {data_dict['capacity']}")

# データセットをファイルからロードする関数
def load_vrp_dataset(filepath):
    """
    VRPデータセットをファイルからロード
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    data = torch.load(filepath, weights_only=False)
    
    print(f"Dataset loaded from: {filepath}")
    return data

# シンプルなVRPデータセットクラス
class SimpleVRPDataset:
    """
    デポ座標、顧客座標、正規化需要を保持するデータセット
    """
    def __init__(self, data_dict):
        self.depot_coords = data_dict['depot_coordinates']
        self.customer_coords = data_dict['customer_coordinates']
        self.norm_demands = data_dict['normalized_demands']
        
        print(f"Creating simple VRP dataset...")
        print(f"Customer Coords: {self.customer_coords.shape}")
    
    def __len__(self):
        return len(self.customer_coords)
    
    def __getitem__(self, idx):
        """1つのインスタンスのデータを返す"""
        return {
            'depot': self.depot_coords[idx],       # (1, 2)
            'customers': self.customer_coords[idx], # (N, 2)
            'demands': self.norm_demands[idx]      # (N,)
        }
    
    def get_batch(self, indices):
        """バッチデータを抽出して返す"""
        return {
            'depot': self.depot_coords[indices],       # (B, 1, 2)
            'customers': self.customer_coords[indices], # (B, N, 2)
            'demands': self.norm_demands[indices]      # (B, N)
        }
    
# シンプルなデータローダークラス
class SimpleVRPDataLoader:
    """VRPデータのバッチを生成するジェネレーター"""
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
    
    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.num_samples)
        else:
            indices = torch.arange(self.num_samples)
        
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.dataset.get_batch(batch_indices)
    
    def __len__(self):
        return self.num_batches
    
# 標準的なデータセットを一括作成する関数
def create_standard_vrp_datasets():
    """標準的な訓練/検証/テスト用VRPデータセットを作成"""
    
    print("=== Creating Standard VRP Datasets ===")
    
    num_cities = 20
    coord_range = 100.0
    
    # 訓練用
    print("\n1. Generating training set...")
    train_data = generate_vrp_data(num_instances=100000, num_cities=num_cities, coord_range=coord_range, seed=42)
    save_vrp_dataset(train_data, f'data/vrp_{num_cities}_train.pt', {'split': 'train'})
    
    # 検証用
    print("\n2. Generating validation set...")
    val_data = generate_vrp_data(num_instances=1000, num_cities=num_cities, coord_range=coord_range, seed=123)
    save_vrp_dataset(val_data, f'data/vrp_{num_cities}_val.pt', {'split': 'val'})
    
    # テスト用
    print("\n3. Generating test set...")
    test_data = generate_vrp_data(num_instances=10000, num_cities=num_cities, coord_range=coord_range, seed=456)
    save_vrp_dataset(test_data, f'data/vrp_{num_cities}_test.pt', {'split': 'test'})
    
    print("\n=== Dataset Creation Complete ===")
    return train_data, val_data, test_data

def test_vrp_dataset_loading():
    """VRPデータセットのロードとバッチ生成のテスト"""
    print("\n=== Testing VRP Dataset Loading ===")
    
    # 1. データのロード
    vrp_data_dict = load_vrp_dataset('data/vrp_20_train.pt')
    
    # 2. DatasetとDataLoaderの準備
    dataset = SimpleVRPDataset(vrp_data_dict[:100])
    dataloader = SimpleVRPDataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"\nデータローダーのテスト:")
    print(f"  総バッチ数: {len(dataloader)}")
    
    # 3. 最初のバッチを取り出して形状を検証
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nバッチ {batch_idx} の検証:")
        
        # 各要素の取り出し
        depot = batch['depot']       # (B, 1, 2)
        customers = batch['customers'] # (B, N, 2)
        demands = batch['demands']     # (B, N)
        
        print(f"  Depot形状:     {depot.shape}")
        print(f"  Customers形状: {customers.shape}")
        print(f"  Demands形状:   {demands.shape}")
        
        # --- 検証 (Assert) ---
        # バッチサイズの一致確認
        assert depot.shape[0] == 16, f"バッチサイズが 16 ではありません"
        
        # 特徴量次元の確認
        assert depot.shape[2] == 2, "デポの座標は (x, y) の2次元である必要があります"
        assert customers.shape[2] == 2, "顧客の座標は (x, y) の2次元である必要があります"
        
        # 都市数(N)の整合性確認
        num_cities = customers.shape[1]
        assert demands.shape[1] == num_cities, "顧客数と需要の数が一致しません"
        
        print(f"  ✓ バッチ {batch_idx} の形状チェック合格")
        
        # テストなので最初のバッチだけで終了
        if batch_idx == 0:
            break
            
    print("\n✓ VRPデータセットのロードテストに成功しました！")