import numpy as np
from scipy.stats import multivariate_normal

class BayesianIdealObserver:
    def __init__(self, params_dict, condition='equal'):
        self.params = params_dict  # Dictionary of mu_noise, mu_weak, mu_medium, mu_strong, sigma
        self.condition = condition  # 'equal' or 'mixture'

    def compute_likelihood(self, x, condition):
        if condition == 'equal':
            return self._likelihood_equal(x)
        elif condition == 'mixture':
            return self._likelihood_mixture(x)

    def _likelihood_equal(self, x):
        mu_signal = np.concatenate([self.params[p]['mu_medium']] * 3)
        mu_noise  = np.concatenate([self.params[p]['mu_noise']] * 3)
        cov = np.eye(9) * self.params['Comp1']['sigma']**2
        return self._likelihood_ratio(x, mu_signal, mu_noise, cov)

    def _likelihood_mixture(self, x):
        likelihoods = []
        for strong_idx in range(3):  # each Comp takes turn being "strong"
            mus = []
            for i, p in enumerate(['Comp1', 'Comp2', 'Comp3']):
                if i == strong_idx:
                    mus.append(self.params[p]['mu_strong'])
                else:
                    mus.append(self.params[p]['mu_weak'])
            mu_signal = np.concatenate(mus)
            mu_noise  = np.concatenate([self.params[p]['mu_noise']] * 3)
            cov = np.eye(9) * self.params['Comp1']['sigma']**2  # assumes same sigma
            likelihoods.append(self._likelihood_ratio(x, mu_signal, mu_noise, cov))
        return np.mean(likelihoods)

    def _likelihood_ratio(self, x, mu_signal, mu_noise, cov):
        p_signal = multivariate_normal.pdf(x, mean=mu_signal, cov=cov)
        p_noise  = multivariate_normal.pdf(x, mean=mu_noise,  cov=cov)
        return p_signal / p_noise

    def predict(self, x):
        lr = self.compute_likelihood(x, self.condition)
        return int(lr > 1)  # 1 = signal present, 0 = absent
