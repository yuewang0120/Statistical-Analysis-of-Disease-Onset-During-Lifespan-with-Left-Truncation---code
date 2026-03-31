import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from utils import Dataset
import matplotlib.pyplot as plt
import os
from utils import trunc_normal_at_3, male_dataset, female_dataset


print("The number of male patients with death: ", male_dataset.data['eta'].sum())
print("The number of male patients with dementia and death: ", ((male_dataset.data['delta']) & (male_dataset.data['eta'])).sum())
print("The number of female patients with death: ", female_dataset.data['eta'].sum())
print("The number of female patients with dementia and death: ", ((female_dataset.data['delta']) & (female_dataset.data['eta'])).sum())

trunc_normal = truncnorm(-3, 3, loc=0, scale=1/3).pdf
l_value = 91
t_value = 102
male_h_value, female_h_value = 2.1, 1.8
s_values = np.arange(l_value, t_value+0.1, 0.1)

male_est, male_se = male_dataset.estimate(s=s_values, t=t_value, l=l_value, h=male_h_value, kernel=trunc_normal, se_fit=True)
female_est, female_se = female_dataset.estimate(s=s_values, t=t_value, l=l_value, h=female_h_value, kernel=trunc_normal, se_fit=True)

plt.figure(figsize=(10, 6))
plt.plot(s_values, male_est, label='Male Estimate', color='red')
plt.fill_between(s_values, male_est - 1.96 * male_se, male_est + 1.96 * male_se, color='orange', alpha=0.3, label='Male 95% CI')
plt.xlabel('s')
plt.ylabel('Probability')
plt.title('Probability Estimates with 95% Confidence Intervals')
plt.legend()
os.makedirs(f'male_l={l_value}_h={male_h_value}', exist_ok=True)
plt.savefig(f'male_l={l_value}_h={male_h_value}/estimate_t={t_value}.png')

plt.figure(figsize=(10, 6))
plt.plot(s_values, female_est, label='Male Estimate', color='red')
plt.fill_between(s_values, female_est - 1.96 * female_se, female_est + 1.96 * female_se, color='orange', alpha=0.3, label='Male 95% CI')
plt.xlabel('s')
plt.ylabel('Probability')
plt.title('Probability Estimates with 95% Confidence Intervals')
plt.legend()
os.makedirs(f'female_l={l_value}_h={female_h_value}', exist_ok=True)
plt.savefig(f'female_l={l_value}_h={female_h_value}/estimate_t={t_value}.png')
