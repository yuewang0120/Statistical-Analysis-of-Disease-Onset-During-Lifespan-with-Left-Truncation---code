import numpy as np
from utils import Design2Dataset as dataset, trunc_normal_at_3
import os
from tqdm import tqdm
from time import time

l_value = 91
n_value = 1000
h_values = np.arange(1, 10, 1)

folder_name = f'Bandwidth_selection_{type(dataset()).__name__}_l={l_value}_n={n_value}'
os.makedirs(folder_name, exist_ok=True)
 

# bandwidth selection
best_bandwidth = []
start_time = time()
for i in tqdm(range(10)):
    sim = dataset()
    sim.sim(n_value, i)
    sim.truncate()
    figure_name = f'{folder_name}/seed={i}.png'
    best_bandwidth.append(sim.bandwidth_selection(5, l_value, h_values, trunc_normal_at_3, figure_name=figure_name))
print("Optimal bandwidth:", best_bandwidth)
print("Average optimal bandwidth:", np.array(best_bandwidth).mean())
print("Average time used:", (time() - start_time) / 10)

# undersmooth
sim = dataset()
sim.sim(100000)
sim.truncate()
sample_size = sim.effective_sample_size(l_value) / 100000 * n_value
print("Effective sample size:", sample_size)
print("Undersmoothed bandwidth:", np.array(best_bandwidth).mean() / sample_size**0.05)


