from utils import Design2Dataset as dataset, trunc_normal_at_3
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

l_value = 91
h_value = 8
n_value = 1000
max_t_value = 107
t_values = np.arange(l_value, max_t_value+0.1, 0.5)
folder_name = f'{type(dataset()).__name__}_l={l_value}_n={n_value}_h={h_value}'
os.makedirs(folder_name, exist_ok=True)

combined_estimates = []
combined_se = []
for i in tqdm(range(1000)):
    sim = dataset()
    sim.sim(n_value, i)
    sim.truncate()
    est, se = sim.estimate_F1t(t=t_values, l=l_value, h=h_value, kernel=trunc_normal_at_3, se_fit=True)
    combined_estimates.append(est)
    combined_se.append(se)
combined_estimates = np.array(combined_estimates)
combined_se = np.array(combined_se)

true_value = sim.true_F1t(t=t_values, l=l_value)
average_estimate = combined_estimates.mean(axis=0)
empirical_upper_bound = average_estimate + 1.96 * combined_estimates.std(axis=0)
empirical_lower_bound = average_estimate - 1.96 * combined_estimates.std(axis=0)
estimated_upper_bound = average_estimate + 1.96 * combined_se.mean(axis=0)
estimated_lower_bound = average_estimate - 1.96 * combined_se.mean(axis=0)


plt.style.use('ggplot')
plt.figure(figsize=(8, 8))
plt.plot(t_values, true_value, label='True Conditional Distribution', color='black')
plt.plot(t_values, average_estimate, label='Average Estimate', linestyle='--', color='black')
plt.plot(t_values, empirical_lower_bound, label='Empirical Lower Bound', linestyle=':', color='black')
plt.plot(t_values, empirical_upper_bound, label='Empirical Upper Bound', linestyle=':', color='black')
plt.plot(t_values, estimated_lower_bound, label='Estimated Lower Bound', linestyle='-.', color='black')
plt.plot(t_values, estimated_upper_bound, label='Estimated Upper Bound', linestyle='-.', color='black')
plt.xlim(l_value-0.5, max_t_value+0.5)
# plt.ylim(-0.05, 0.6)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(f'{folder_name}/F1t_estimate.png')

coverage_prob = ((combined_estimates + 1.96 * combined_se > true_value) & (combined_estimates - 1.96 * combined_se < true_value)).mean(axis=0)
plt.figure(figsize=(8, 8))
plt.plot(t_values, coverage_prob, label='Coverage Probability', color='black')
plt.axhline(y=0.95, linestyle=':', label='95% Coverage', color='black')
plt.xlim(l_value-0.5, max_t_value+0.5)
plt.ylim(-0.05, 1.05)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(f'{folder_name}/F1t_coverage.png')
