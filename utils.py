import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, expon, beta, norm, truncnorm
from scipy.integrate import quad
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

class Dataset:
    def __init__(self, data=None):
        self.data = data

    def name(self, name):
        self.name = name

    def estimate_F1s_bspline(self, s, t, l, h, kernel, se_fit=False, scale='plain', boostrap_n=100, k=3):

        mask = (self.data['eta'] == 1) & (self.data['delta'] == 1) & (self.data['Y'] >= l)
        Y_selected = np.unique(self.data.loc[mask, 'Y'].to_numpy())

        est, se = self.estimate_F1s(Y_selected, t, l, h, kernel, se_fit=False, scale=scale)

        Y_selected = np.concatenate(([l], Y_selected))
        est = np.concatenate(([0], est))

        est_bspline = make_interp_spline(Y_selected, est, k=k)(s)

        if se_fit:
            for _boot in range(boostrap_n):
                sample_indices = np.random.choice(len(self.data), len(self.data), replace=True)
                data_bootstrap = Dataset(self.data.iloc[sample_indices].reset_index(drop=True))
                est_bootstrap, _ = data_bootstrap.estimate_F1s_bspline(s, t, l, h, kernel, se_fit=False, scale=scale, k=k)
                if _boot == 0:
                    bootstrap_estimates = est_bootstrap[:, None]
                else:
                    bootstrap_estimates = np.hstack((bootstrap_estimates, est_bootstrap[:, None]))
            return est_bspline, np.std(bootstrap_estimates, axis=1)
        
        else:
            return est_bspline, None

    def estimate_F1s(self, s, t, l, h, kernel, se_fit=False, scale='plain'):
        est = np.ones_like(s)
        var = np.zeros_like(s)

        X = self.data['X'].to_numpy()
        Y = self.data['Y'].to_numpy()
        L = self.data['L'].to_numpy()
        eta = self.data['eta'].to_numpy()
        delta = self.data['delta'].to_numpy()

        K = kernel((X - t) / h)

        if se_fit:
            # x = np.arange(-1, 1, 0.01)
            # y = kernel(x)
            # mu1 = np.trapz(y, x)
            # mu2 = np.trapz(y ** 2, x)
            mu1 = kernel.mu1
            mu2 = kernel.mu2

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

        F1s = 1 - est
            
        if scale == 'plain':
            return F1s, np.sqrt(mu2 / mu1 * var) * (1 - F1s) if se_fit else None
        if scale == 'log':
            return -np.log(1 - F1s), np.sqrt(mu2 / mu1 * var) if se_fit else None
        if scale == 'cll':
            return np.log(-np.log(1 - F1s)), np.sqrt(mu2 / mu1 * var) / -np.log(1 - F1s) if se_fit else None
        
    def estimate_F1t(self, t, l, h, kernel, se_fit=False, scale='plain'):
        est = []
        se = []
        for t_i in t:
            est_i, se_i = self.estimate_F1s(t_i[None], t_i, l, h, kernel, se_fit, scale=scale)
            est.append(est_i)
            se.append(se_i)
        return np.concatenate(est), np.concatenate(se) if se_fit else None
        
    def estimate_F2s(self, s, t, l, h, kernel, se_fit=False, scale='plain'):
        F1s, F1s_se = self.estimate_F1s(s, t, l, h, kernel, se_fit, scale='plain')
        F1t, F1t_se = self.estimate_F1s(np.array([t]), t, l, h, kernel, se_fit, scale='plain')
        F2s = F1s / F1t
        if se_fit:
            a = (1 + F1s - 2 * F2s) / F1t**2 / (1 - F1s)
            b = F2s**2 / F1t**2
            var = a * F1s_se**2 + b * F1t_se**2
            var = np.where((var < 0) & np.isclose(var, 0), 0, var)
        if scale == 'plain':
            return F2s, np.sqrt(var) if se_fit else None
        if scale == 'log':
            return -np.log(1 - F2s), np.sqrt(var) / (1 - F2s) if se_fit else None
        if scale == 'cll':
            return np.log(-np.log(1 - F2s)), np.sqrt(var) / (1 - F2s) / -np.log(1 - F2s) if se_fit else None
    
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
# trunc_normal_at_3 = truncnorm(-3, 3, loc=0, scale=1).pdf

class Kernel:
    """
    Wrap a kernel/pdf callable and precompute its mu1 = ∫ K(u) du and mu2 = ∫ K(u)^2 du.
    The provided pdf should accept numpy arrays (works elementwise). If it only accepts
    scalars, the class will still work because scipy.quad passes scalars to the pdf.
    """
    def __init__(self, pdf, support=None, grid_bounds=10, grid_n=20001, atol=1e-8):
        self.pdf = pdf  # callable: pdf(x) -> array_like
        self.support = support

        # compute mu1 and mu2
        if support is None:
            try:
                self.mu1, _ = quad(lambda u: float(self.pdf(u)), -np.inf, np.inf, epsabs=atol)
                self.mu2, _ = quad(lambda u: float(self.pdf(u))**2, -np.inf, np.inf, epsabs=atol)
            except Exception:
                # fallback to a wide finite grid if quad fails
                xs = np.linspace(-grid_bounds, grid_bounds, grid_n)
                ys = np.asarray(self.pdf(xs), dtype=float)
                self.mu1 = np.trapz(ys, xs)
                self.mu2 = np.trapz(ys**2, xs)
        else:
            a, b = support
            xs = np.linspace(a, b, grid_n)
            ys = np.asarray(self.pdf(xs), dtype=float)
            self.mu1 = np.trapz(ys, xs)
            self.mu2 = np.trapz(ys**2, xs)

    def __call__(self, x):
        x_arr = np.asarray(x)
        return self.pdf(x_arr)

    @classmethod
    def gaussian(cls, sigma=1.0):
        return cls(lambda u: np.exp(-0.5 * (u / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma))

    @classmethod
    def epanechnikov(cls):
        def k(u):
            u = np.asarray(u)
            out = np.zeros_like(u, dtype=float)
            mask = np.abs(u) <= 1
            out[mask] = 3.0 / 4.0 * (1 - u[mask] ** 2)
            return out
        return cls(k, support=(-1, 1))

    @classmethod
    def uniform(cls):
        def k(u):
            u = np.asarray(u)
            out = np.zeros_like(u, dtype=float)
            out[np.abs(u) <= 1] = 0.5
            return out
        return cls(k, support=(-1, 1))

    @classmethod
    def triangular(cls):
        def k(u):
            u = np.asarray(u)
            out = np.zeros_like(u, dtype=float)
            mask = np.abs(u) <= 1
            out[mask] = 1 - np.abs(u[mask])
            return out
        return cls(k, support=(-1, 1))
    
trunc_normal_at_3 = Kernel(truncnorm(-3, 3, loc=0, scale=1).pdf)
# print(trunc_normal_at_3.mu1, trunc_normal_at_3.mu2)

df = pd.read_excel('nan01_req0423.xlsx')
df['L'] = df['age_first']
df['X'] = df['agedeath'].combine_first(df['age_last'])
df['eta'] = ~df['agedeath'].isna()
df['Y'] = df['age_last']
df['delta'] = df['dement_last'] == 1
male_dataset = Dataset(df[df['Gender'] == 0])
female_dataset = Dataset(df[df['Gender'] == 1])
