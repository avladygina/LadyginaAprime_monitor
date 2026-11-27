import numpy as np
from scipy import stats


def n_per_group(sigma, mde, alpha=0.05, power=0.8):
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * (z * sigma / mde) ** 2))


def mde_for_n(sigma, n, alpha=0.05, power=0.8):
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * sigma * np.sqrt(2 / n)


def diff_ci(t, c, alpha=0.05):
    t, c = np.asarray(t, float), np.asarray(c, float)
    diff = t.mean() - c.mean()
    vt, vc = t.var(ddof=1) / len(t), c.var(ddof=1) / len(c)
    se = np.sqrt(vt + vc)
    df = (vt + vc) ** 2 / (vt**2 / (len(t) - 1) + vc**2 / (len(c) - 1))
    q = stats.t.ppf(1 - alpha / 2, df)
    p = stats.ttest_ind(t, c, equal_var=False).pvalue
    return {"diff": diff, "lo": diff - q * se, "hi": diff + q * se, "p": p}


def srm_pvalue(sizes):
    return stats.chisquare(np.asarray(sizes, float)).pvalue


def cuped(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    theta = np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)
    return y - theta * (x - x.mean()), theta