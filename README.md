# Structure-As-Search

> A PyTorch implementation of **Structure-As-Search**, a fully unsupervised, non-autoregressive framework for solving the Traveling Salesman Problem (TSP).


## Installation & Dependencies

- ### Required Dependencies
Before running the code, ensure you have installed the torch-linear-assignment: https://www.piwheels.org/project/torch-linear-assignment. I am using Version 0.0.3.




- ### Important Training Note
⚠️ **Mixed precision training can cause problems during inference.** If you encounter issues during inference, consider disabling mixed precision training.


## TSP-50 (default) Demonstration

This example demonstrates inference on 50-node TSP instances using the **Structure-As-Search** model.

The ensemble method is based on 20 coprime shifts of the cyclic Hamiltonian matrix `V^k`, where each `--shift_-k` corresponds to a coprime `k` such that `gcd(k, 50) = 1`:

- `--shift_-1`  →  coprime 1  
- `--shift_-3`  →  coprime 3  
- ...  
- `--shift_-49` →  coprime 49  

Each coprime defines a distinct Hamiltonian cycle topology used to train a separate model variant.

- ### Training

To train on 50-node TSP instances, use the provided SLURM batch script:

```bash
Train50.sh
```
----
- ### Inference Command
Run inference using a trained model (e.g., for V):
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.3_shift_-1_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

Run inference using a trained model (e.g., for V^3):
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.05_shift_-3_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```
<details>
  <summary>Click to expand more shifts on TSP-50 (w.r.t. V^k):
</summary>
  
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.005_shift_-7_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```
```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.05_shift_-9_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.01_shift_-11_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.005_shift_-13_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.2_shift_-17_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.2_shift_-19_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.01_shift_-21_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.05_shift_-23_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.05_shift_-27_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.05_shift_-29_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.05_shift_-31_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.1_shift_-33_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.1_shift_-37_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.05_shift_-39_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.01_shift_-41_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.005_shift_-43_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.2_shift_-47_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

```
python test.py   --test_data data/tsp_50_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_50_hidden_256_adam_tau_5.0_n_iter_80_noise_0.005_shift_-49_dist_scale_5.0_n_layers6_seed_42.pt --num_nodes 50 --compute_greedy
```

</details>

## TSP-100


- ### Training

To train on 100-node TSP instances, use the provided SLURM batch script:

```bash
Train100.sh
```
----

- ### Inference Command
 

<details>
  <summary>Click to expand more shifts on TSP-100 (w.r.t. V^k):
</summary>

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.1_shift_-1_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100 
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.005_shift_-3_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100 
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.3_shift_-7_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100 
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.2_shift_-9_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.01_shift_-11_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100 
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.02_shift_-13_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.03_shift_-17_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100 
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.01_shift_-19_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.02_shift_-21_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100 
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.05_shift_-23_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.05_shift_-27_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.02_shift_-29_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100 
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.005_shift_-31_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.005_shift_-33_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.1_shift_-37_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.2_shift_-39_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.02_shift_-41_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100 
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.005_shift_-43_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.005_shift_-47_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.02_shift_-49_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.3_shift_-51_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.1_shift_-53_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100 
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.01_shift_-57_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.2_shift_-59_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.3_shift_-61_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.3_shift_-63_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.3_shift_-67_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100 
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.005_shift_-69_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.05_shift_-71_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.2_shift_-73_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.02_shift_-77_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.05_shift_-79_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100 
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.2_shift_-81_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100 
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.01_shift_-83_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.01_shift_-87_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.01_shift_-89_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100 
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.2_shift_-91_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100 
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.3_shift_-93_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.03_shift_-97_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100
```

```
python test.py   --test_data data/tsp_100_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_100_hidden_512_adam_tau_5.0_n_iter_80_noise_0.01_shift_-99_dist_scale_5.0_n_layers4_seed_42.pt --num_nodes 100 
```


</details>



## TSP-20


- ### Training

To train on 20-node TSP instances, use the provided SLURM batch script:

```bash
Train20.sh
```
----

- ### Inference Command


<details>
  <summary>Click to expand more shifts on TSP-100 (w.r.t. V^k):
</summary>

```
python test.py   --test_data data/tsp_20_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_20_hidden_128_adam_tau_2.0_n_iter_60_noise_0.05_shift_-1_dist_scale_5.0_n_layers2_seed_42.pt --num_nodes 20
```

```
python test.py   --test_data data/tsp_20_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_20_hidden_128_adam_tau_2.0_n_iter_60_noise_0.05_shift_-3_dist_scale_5.0_n_layers2_seed_42.pt --num_nodes 20
```

```
python test.py   --test_data data/tsp_20_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_20_hidden_128_adam_tau_2.0_n_iter_60_noise_0.01_shift_-7_dist_scale_5.0_n_layers2_seed_42.pt --num_nodes 20
```

```
python test.py   --test_data data/tsp_20_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_20_hidden_128_adam_tau_2.0_n_iter_60_noise_0.3_shift_-9_dist_scale_5.0_n_layers2_seed_42.pt --num_nodes 20 
```

```
python test.py   --test_data data/tsp_20_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_20_hidden_128_adam_tau_2.0_n_iter_60_noise_0.3_shift_-11_dist_scale_5.0_n_layers2_seed_42.pt --num_nodes 20 
```

```
python test.py   --test_data data/tsp_20_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_20_hidden_128_adam_tau_2.0_n_iter_60_noise_0.01_shift_-13_dist_scale_5.0_n_layers2_seed_42.pt  --num_nodes 20
```

```
python test.py   --test_data data/tsp_20_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_20_hidden_128_adam_tau_2.0_n_iter_60_noise_0.05_shift_-17_dist_scale_5.0_n_layers2_seed_42.pt --num_nodes 20
```

```
python test.py   --test_data data/tsp_20_uniform_test.pt --save_dir test_results_shift --model_path SaveModels/best_stable_sct_model_size_20_hidden_128_adam_tau_2.0_n_iter_60_noise_0.05_shift_-19_dist_scale_5.0_n_layers2_seed_42.pt  --num_nodes 20
```


</details>



## Run Hamiltonian Cycle Ensemble

go to the dir which saves the inference results

```
cd test_results_shift
```

For TSP-50, run

```
python HamiltonianCycleEnsemble.py all_tour_lengths_shift_-{1,3,7,9,11,13,17,19,21,23,27,29,31,33,37,39,41,43,47,49}_size_50.txt
```
For TSP-100, run
```
python HamiltonianCycleEnsemble.py all_tour_lengths_shift_-{1,3,7,9,11,13,17,19,21,23,27,29,31,33,37,39,41,43,47,49,51,53,57,59,61,63,67,69,71,73,77,79,81,83,87,89,91,93,97,99}_size_100.txt
```
For TSP-20, run
```
python HamiltonianCycleEnsemble.py  all_tour_lengths_shift_-{1,3,7,9,11,13,17,19}_size_20.txt
```

-----


some code are adapted from utsp: https://github.com/yimengmin/UTSP and ordering clique: https://github.com/yimengmin/UnsupervisedOrderingMaximumClique
