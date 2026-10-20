from scipy.stats import norm, chisquare
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
import numpy as np

def sample_size(p_base, mde_real, k=0.85, alpha = 0.05, power = 0.8):
    z_a = norm.ppf(1 - alpha/2)
    z_b = norm.ppf(power)
    return p_base * (1 - p_base) * (1/k + 1/(1-k)) * (z_a + z_b) ** 2 / (p_base * mde_real) ** 2

def sample_size_statsmodels(p_base, mde_rel, k=0.85, alpha=0.05, power=0.8):
    p_new = p_base * (1 + mde_rel)
    effect = proportion_effectsize(p_new, p_base)   
    ratio = k / (1 - k)                            
    n_control = NormalIndPower().solve_power(
        effect_size=effect,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative="two-sided",
    )
    return int(np.ceil(n_control * (1 + ratio)))

def mde_value(N, p_base, k=0.85, alpha=0.05, power=0.8):
    z_a = norm.ppf(1 - alpha/2)
    z_b = norm.ppf(power)
    return ((p_base * (1 - p_base) * (1/k + 1/(1-k)) * (z_a + z_b) ** 2 / N) ** 0.5) / p_base

def srm_check(n_treatment, n_control, expected_share=0.85, threshold=0.001):
    n_total = n_treatment + n_control
    observed = [n_treatment, n_control]
    expected = [n_total * expected_share, n_total * (1 - expected_share)]
    stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    return stat, p_value, p_value < threshold

def ztest_props(p_t, n_t, p_c, n_c, alpha=0.05):
    diff = p_t - p_c
    p_pool = (p_t * n_t + p_c * n_c) / (n_t + n_c)
    se = (p_pool * (1 - p_pool) * (1/n_c + 1/n_t)) ** 0.5
    z_crit = norm.ppf(1 - alpha/2)
    z = diff / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return {
        "p_t": p_t,
        "p_c": p_c,
        "diff": diff,
        "se": se,
        "ci": (diff - z_crit * se, diff + z_crit * se),
        "z": z,
        "p_value": p_value,
    }