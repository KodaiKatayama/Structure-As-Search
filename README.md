# Structure-As-Search

> A PyTorch implementation of **Structure-As-Search**, a fully unsupervised, non-autoregressive framework for solving the Traveling Salesman Problem (TSP).
> **Paper**: *Structure As Search: Unsupervised Permutation Learning for Combinatorial Optimization*  
> **Authors**: Yimeng Min, Carla P. Gomes


##  TSP-50 Inference Demonstration

This example demonstrates inference on 50-node TSP instances using the **Structure-As-Search** model.

The ensemble method is based on 20 coprime shifts of the cyclic Hamiltonian matrix `V^k`, where each `--shift_-k` corresponds to a coprime `k` such that `gcd(k, 50) = 1`:

- `--shift_-1`  →  coprime 1  
- `--shift_-3`  →  coprime 3  
- ...  
- `--shift_-49` →  coprime 49  

Each coprime defines a distinct Hamiltonian cycle topology used to train a separate model variant.

### Training

To train on 50-node TSP instances, use the provided SLURM batch script:

```bash
Train50.sh
```
----
### Inference Command
Run inference using a trained model (e.g., for V):
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_3.0_n_iter_80_noise_0.1_shift_-1_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50
```

Run inference using a trained model (e.g., for V^3):
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_3.0_n_iter_80_noise_0.005_shift_-3_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_3.0_n_iter_80_noise_0.2_shift_-7_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50 
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_3.0_n_iter_80_noise_0.2_shift_-9_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50 
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_4.0_n_iter_80_noise_0.1_shift_-11_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_3.0_n_iter_80_noise_0.1_shift_-13_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50 
```
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_4.0_n_iter_80_noise_0.005_shift_-17_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50 
```
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_4.0_n_iter_80_noise_0.005_shift_-19_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50 
```
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_4.0_n_iter_80_noise_0.1_shift_-21_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50 
```
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_4.0_n_iter_80_noise_0.01_shift_-23_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50 
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_3.0_n_iter_80_noise_0.01_shift_-27_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50 
```
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.005_shift_-29_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50 
```
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.01_shift_-31_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50 
```
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_4.0_n_iter_80_noise_0.2_shift_-33_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50 
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_3.0_n_iter_80_noise_0.05_shift_-37_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50 
```
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_4.0_n_iter_80_noise_0.05_shift_-39_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50 
```
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_4.0_n_iter_80_noise_0.1_shift_-41_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50 
```
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_3.0_n_iter_80_noise_0.05_shift_-43_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50 
```
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_3.0_n_iter_80_noise_0.01_shift_-47_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50 
```
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_3.0_n_iter_80_noise_0.01_shift_-49_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 50 
```
