from utils import Design1Dataset as dataset, trunc_normal_at_3
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

l_value = 91
h_value = 4
n_value = 1000
nrep = 1000
scale = 'log'
folder_name = f'{type(dataset()).__name__}_l={l_value}_n={n_value}_h={h_value}_nrep={nrep}_scale={scale}'
os.makedirs(folder_name, exist_ok=True)
def reverse_cll_transform(x):
    return 1 - np.exp(-np.exp(x))
def reverse_log_transform(x):
    return 1 - np.exp(-x)

for t_value in [95, 100, 105]:
    
    s_values = np.arange(l_value, t_value+0.1, 0.1)
    
    combined_estimates = []
    combined_se = []
    for i in tqdm(range(nrep)):
        sim = dataset()
        sim.sim(n_value, i)
        sim.truncate()
        est, se = sim.estimate_F1s(s=s_values, t=t_value, l=l_value, h=h_value, kernel=trunc_normal_at_3, se_fit=True, scale=scale)
        combined_estimates.append(est)
        combined_se.append(se)
    combined_estimates = np.array(combined_estimates)
    combined_se = np.array(combined_se)

    true_value = sim.true_F1s(s=s_values, t=t_value, l=l_value)
    average_estimate = combined_estimates.mean(axis=0)
    empirical_upper_bound = average_estimate + 1.96 * combined_estimates.std(axis=0)
    empirical_lower_bound = average_estimate - 1.96 * combined_estimates.std(axis=0)
    estimated_upper_bound = average_estimate + 1.96 * combined_se.mean(axis=0)
    estimated_lower_bound = average_estimate - 1.96 * combined_se.mean(axis=0)


    print(f"Plotting F1 for t={t_value}")
    plt.style.use('ggplot')
    plt.figure(figsize=(8, 8))
    plt.plot(s_values, true_value, label='True Conditional Distribution', color='black')
    plt.plot(s_values, average_estimate, label='Average Estimate', linestyle='--', color='black')
    plt.plot(s_values, empirical_lower_bound, label='Empirical Lower Bound', linestyle=':', color='black')
    plt.plot(s_values, empirical_upper_bound, label='Empirical Upper Bound', linestyle=':', color='black')
    plt.plot(s_values, estimated_lower_bound, label='Estimated Lower Bound', linestyle='-.', color='black')
    plt.plot(s_values, estimated_upper_bound, label='Estimated Upper Bound', linestyle='-.', color='black')
    plt.xlim(l_value-0.25, t_value+0.25)
    plt.ylim(-0.05, 1.05)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{folder_name}/F1s_estimate_t={t_value}.png')

    print(f"Plotting coverage of F1 for t={t_value}")
    
    if scale == 'plain':
        coverage_prob = ((combined_estimates + 1.96 * combined_se > true_value) & (combined_estimates - 1.96 * combined_se < true_value)).mean(axis=0)
    elif scale == 'log':
        coverage_prob = ((reverse_log_transform(combined_estimates + 1.96 * combined_se) > true_value) & (reverse_log_transform(combined_estimates - 1.96 * combined_se) < true_value)).mean(axis=0)
    elif scale == 'cll':
        coverage_prob = ((reverse_cll_transform(combined_estimates + 1.96 * combined_se) > true_value) & (reverse_cll_transform(combined_estimates - 1.96 * combined_se) < true_value)).mean(axis=0)
    plt.figure(figsize=(8, 8))
    plt.plot(s_values, coverage_prob, label='Coverage Probability', color='black')
    plt.axhline(y=0.95, linestyle=':', label='95% Coverage', color='black')
    plt.xlim(l_value-0.25, t_value+0.25)
    plt.ylim(-0.05, 1.05)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{folder_name}/F1s_coverage_t={t_value}.png')
