import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, expon, beta, norm, truncnorm

class Dataset:
    def __init__(self, data=None):
        self.data = data

    def name(self, name):
        self.name = name

    def estimate_F1s(self, s, t, l, h, kernel, se_fit=False):
        est = np.ones_like(s)
        var = np.zeros_like(s)

        X = self.data['X'].to_numpy()
        Y = self.data['Y'].to_numpy()
        L = self.data['L'].to_numpy()
        eta = self.data['eta'].to_numpy()
        delta = self.data['delta'].to_numpy()

        K = kernel((X - t) / h)

        if se_fit:
            x = np.arange(-1, 1, 0.01)
            y = kernel(x)
            mu1 = np.trapz(y, x)
            mu2 = np.trapz(y ** 2, x)

        for i in range(self.data.shape[0]):
            if eta[i] == 0 or l > Y[i] or delta[i] == 0:
                continue
            
            y_i = Y[i]

            numerator = (s >= y_i) * K[i]
            denominator = (eta * (L <= y_i) * (Y >= y_i) * K).sum()

            if denominator > 0:
                est = est * (1 - numerator / denominator)
                if se_fit:
                    var = var + numerator / denominator**2
            
        if se_fit:
            return 1 - est, np.sqrt(mu2 / mu1 * var) * est
        else:
            return 1 - est
        
    def estimate_F1t(self, t, l, h, kernel, se_fit=False):
        est = []
        se = []
        for t_i in t:
            est_i, se_i = self.estimate_F1s(t_i[None], t_i, l, h, kernel, se_fit)
            est.append(est_i)
            se.append(se_i)
        return np.concatenate(est), np.concatenate(se)
        
    def estimate_F2s(self, s, t, l, h, kernel, se_fit=False):
        if se_fit:
            F1s, F1s_se = self.estimate_F1s(s, t, l, h, kernel, se_fit)
            F1t, F1t_se = self.estimate_F1s(np.array([t]), t, l, h, kernel, se_fit)
            F2s = F1s / F1t
            a = (1 + F1s - 2 * F2s) / F1t**2 / (1 - F1s)
            b = F2s**2 / F1t**2
            var = a * F1s_se**2 + b * F1t_se**2
            var = np.where((var < 0) & np.isclose(var, 0), 0, var)
            return F2s, np.sqrt(var)
        else:
            F1s = self.estimate_F1s(s, t, l, h, kernel, se_fit)
            F1t = self.estimate_F1s(np.array([t]), t, l, h, kernel, se_fit)
            F2s = F1s / F1t
            return F2s

    
    def k_fold_split(self, k):
        # yield a pair of train and test data of class Dataset for each fold
        folds = np.array_split(self.data.sample(frac=1, random_state=42), k)
        for i in range(k):
            train_data = pd.concat([folds[j] for j in range(k) if j != i])
            test_data = folds[i]
            yield Dataset(train_data), Dataset(test_data)
    
    def min_S_above_cutoff(self, cutoff):
        return self.data['Y'][self.data['delta'] & self.data['eta'] & (self.data['Y']>cutoff)].min()
    
    def max_S_under_cutoff(self, cutoff):
        return self.data['Y'][self.data['delta'] & self.data['eta'] & (self.data['Y']<cutoff)].max()

    def max_X_among_complete(self):
        return self.data['X'][self.data['eta']].max()
    
    def k_fold_cv(self, k, l, h, kernel):
        score = 0
        for train_data, test_data in self.k_fold_split(k):
            # s_values = np.arange(l, test_data.max_X_among_complete(), 0.1) # this version is not good
            s_values = np.arange(test_data.min_S_above_cutoff(l), test_data.max_S_under_cutoff(np.inf), 0.1)
            numerator = np.zeros_like(s_values)
            denominator = np.zeros_like(s_values)
            for i in range(len(test_data.data)):
                x_i = test_data.data['X'].iloc[i]
                y_i = test_data.data['Y'].iloc[i]
                eta_i = test_data.data['eta'].iloc[i]
                
                if x_i <= s_values[0] or y_i < l:
                    continue  # skip this sample entirely based on logic in your vectorized version
                
                F_hat_i = train_data.estimate_F1s(s_values, x_i, l, h, kernel)  # shape (len(s_values),)
                
                # Conditions applied per s
                mask = (x_i > s_values)  # shape (len(s_values),)
                y_in_range = (y_i <= s_values)  # shape (len(s_values),)

                err_sq = ((y_in_range.astype(float) - F_hat_i) ** 2)  # shape (len(s_values),)
                
                numerator += eta_i * mask * (y_i >= l) * err_sq
                denominator += eta_i * mask * (y_i >= l)
            
            # Compute Brier score
            brier_score = numerator / denominator
            integrated_brier_score = np.trapz(brier_score, s_values) / (s_values.max() - s_values.min())
            score += integrated_brier_score
        return score / k
    
    def bandwidth_selection(self, k, l, hlist, kernel, figure_name=None):
        scores = []
        for h in hlist:
            scores.append(self.k_fold_cv(k, l, h, kernel))
        plt.figure(figsize=(10, 6))
        plt.plot(hlist, scores)
        plt.xlabel('Bandwidth')
        plt.ylabel('Integrated Brier Score')
        plt.title('Integrated Brier Score vs Bandwidth')
        if figure_name:
            plt.savefig(figure_name)
        else:
            plt.show()
        return hlist[np.argmin(scores)]
    
    def effective_sample_size(self, l):
        return self.data['eta'][self.data['Y'] >= l].sum()
    
class Design1Dataset(Dataset):
    def sim(self, n, seed=None):
        np.random.seed(seed) 
        mean = np.array([0, 0, 0])
        cov = np.array([[1, 0.5, 0.5],
                        [0.5, 1, 0.5],
                        [0.5, 0.5, 1]]) 
        TCL = multivariate_normal.rvs(mean, cov, size=n)
        
        T = expon.ppf(norm.cdf(TCL[:, 0])) * 10 + 90
        L = norm.cdf(TCL[:, 1]) * 5 + 90
        C = expon.ppf(norm.cdf(TCL[:, 2])) * 20 + 90
        S = beta.rvs(0.2 * T - 17, 1, size=n) * 30 + 80

        self.data = pd.DataFrame({
            'L': L, 'T': T, 'C': C, 'S': S
        })
        self.data['X'] = np.minimum(self.data['T'], self.data['C'])
        self.data['eta'] = self.data['T'] <= self.data['C']
        self.data['Y'] = np.minimum.reduce([self.data['S'], self.data['X']])
        self.data['delta'] = self.data['S'] <= self.data['X']
    
    
    def truncate(self):
        self.data = self.data[(self.data['Y'] >= self.data['L'])]

    def true_dist(self, s, t):
        return beta.cdf((s - 80) / 30, 0.2 * t - 17, 1)

    def true_F1s(self, s, t, l):
        return np.clip((self.true_dist(s, t) - self.true_dist(l, t)) / (1 - self.true_dist(l, t)), 0, 1)
    
    def true_F2s(self, s, t, l):
        return self.true_F1s(s, t, l) / self.true_F1s(t, t, l)
    
    def true_F1t(self, t, l):
        return [self.true_F1s(t_i, t_i, l) for t_i in t]
    
class Design2Dataset(Dataset):
    def sim(self, n, seed=None):
        np.random.seed(seed) 
        mean = np.array([0, 0, 0])
        cov = np.array([[1, 0.5, 0.5],
                        [0.5, 1, 0.5],
                        [0.5, 0.5, 1]]) 
        TCL = multivariate_normal.rvs(mean, cov, size=n)
        
        T = expon.ppf(norm.cdf(TCL[:, 0])) * 10 + 90
        L = expon.ppf(norm.cdf(TCL[:, 1])) * 3 + 90
        C = expon.ppf(norm.cdf(TCL[:, 2])) * 20 + 90
        S = beta.rvs(0.2 * T - 17, 2, size=n) * 30 + 80

        self.data = pd.DataFrame({
            'L': L, 'T': T, 'C': C, 'S': S
        })
        self.data['X'] = np.minimum(self.data['T'], self.data['C'])
        self.data['eta'] = self.data['T'] <= self.data['C']
        self.data['Y'] = np.minimum.reduce([self.data['S'], self.data['X']])
        self.data['delta'] = self.data['S'] <= self.data['X']
    
    
    def truncate(self):
        self.data = self.data[(self.data['Y'] >= self.data['L'])]

    def true_dist(self, s, t):
        return beta.cdf((s - 80) / 30, 0.2 * t - 17, 2)

    def true_F1s(self, s, t, l):
        return np.clip((self.true_dist(s, t) - self.true_dist(l, t)) / (1 - self.true_dist(l, t)), 0, 1)

    def true_F2s(self, s, t, l):
        return self.true_F1s(s, t, l) / self.true_F1s(t, t, l)
    
    def true_F1t(self, t, l):
        return [self.true_F1s(t_i, t_i, l) for t_i in t]

class Design3Dataset(Dataset):
    def sim(self, n, seed=None):
        np.random.seed(seed) 
        mean = np.array([0, 0, 0])
        cov = np.array([[1, 0.5, 0.5],
                        [0.5, 1, 0.5],
                        [0.5, 0.5, 1]]) 
        TCL = multivariate_normal.rvs(mean, cov, size=n)
        
        T = expon.ppf(norm.cdf(TCL[:, 0])) * 10 + 90
        L = expon.ppf(norm.cdf(TCL[:, 1])) * 3 + 90
        C = expon.ppf(norm.cdf(TCL[:, 2])) * 8 + 95
        S = beta.rvs(0.2 * T - 17, 2, size=n) * 30 + 80

        self.data = pd.DataFrame({
            'L': L, 'T': T, 'C': C, 'S': S
        })
        self.data['X'] = np.minimum(self.data['T'], self.data['C'])
        self.data['eta'] = self.data['T'] <= self.data['C']
        self.data['Y'] = np.minimum.reduce([self.data['S'], self.data['X']])
        self.data['delta'] = self.data['S'] <= self.data['X']
    
    
    def truncate(self):
        self.data = self.data[(self.data['Y'] >= self.data['L'])]

    def true_dist(self, s, t):
        return beta.cdf((s - 80) / 30, 0.2 * t - 17, 2)

    def true_F1s(self, s, t, l):
        return np.clip((self.true_dist(s, t) - self.true_dist(l, t)) / (1 - self.true_dist(l, t)), 0, 1)
    
    def true_F2s(self, s, t, l):
        return self.true_F1s(s, t, l) / self.true_F1s(t, t, l)
    
    def true_F1t(self, t, l):
        return [self.true_F1s(t_i, t_i, l) for t_i in t]


    
# trunc_normal_at_3 = truncnorm(-3, 3, loc=0, scale=1/3).pdf
trunc_normal_at_3 = truncnorm(-3, 3, loc=0, scale=1).pdf

df = pd.read_excel('nan01_req0423.xlsx')
df['L'] = df['age_first']
df['X'] = df['agedeath'].combine_first(df['age_last'])
df['eta'] = ~df['agedeath'].isna()
df['Y'] = df['age_last']
df['delta'] = df['dement_last'] == 1
male_dataset = Dataset(df[df['Gender'] == 0])
female_dataset = Dataset(df[df['Gender'] == 1])
