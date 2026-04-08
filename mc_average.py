import numpy as np


def _autocovariance_unbiased(x: np.ndarray) -> np.ndarray:
    n = x.size
    x = x - np.mean(x)
    if n == 0:
        return x
    fft = np.fft.rfft(x, n=2 * n)
    acov = np.fft.irfft(fft * np.conjugate(fft))[:n]
    acov /= np.arange(n, 0, -1)
    return acov


def mc_average(data, max_lag=None, c=5.0):
    """
    Estimate mean and uncertainty for correlated MC samples.

    Returns:
        mean, stderr, tau_int, n_eff, autocorr
    """
    x = np.asarray(data, dtype=float)
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0.0, np.array([])
    if n == 1:
        return float(x[0]), 0.0, 0.5, 1.0, np.array([1.0])

    acov = _autocovariance_unbiased(x)
    var = acov[0]
    if var == 0.0:
        return float(np.mean(x)), 0.0, 0.5, float(n), np.ones(n)

    autocorr = acov / var
    if max_lag is None:
        max_lag = n - 1
    max_lag = min(max_lag, n - 1)

    tau_int = 0.5
    for t in range(1, max_lag + 1):
        if autocorr[t] <= 0.0:
            break
        tau_int = 0.5 + np.sum(autocorr[1 : t + 1])
        if t > c * tau_int:
            break

    n_eff = n / (2.0 * tau_int) if tau_int > 0 else float(n)
    stderr = np.sqrt(var * 2.0 * tau_int / n)
    return float(np.mean(x)), float(stderr), float(tau_int), float(n_eff), autocorr


# def _ar1_process(n, rho, rng):
#     x = np.zeros(n, dtype=float)
#     sigma = np.sqrt(1.0 - rho ** 2)
#     for i in range(1, n):
#         x[i] = rho * x[i - 1] + sigma * rng.normal()
#     return x


# if __name__ == "__main__":
#     rng = np.random.default_rng(0)

#     n = 20000
#     x_uncorr = rng.normal(size=n)
#     mean, err, tau_int, n_eff, _ = mc_average(x_uncorr)
#     print("Uncorrelated samples")
#     print("mean:", mean, "stderr:", err, "tau_int:", tau_int, "n_eff:", n_eff)

#     rho = 0.8
#     x_corr = _ar1_process(n, rho, rng)
#     mean, err, tau_int, n_eff, _ = mc_average(x_corr)
#     tau_theory = 0.5 + rho / (1.0 - rho)
#     print("\nAR(1) correlated samples")
#     print("rho:", rho, "tau_int (est):", tau_int, "tau_int (theory):", tau_theory)
#     print("mean:", mean, "stderr:", err, "n_eff:", n_eff)
