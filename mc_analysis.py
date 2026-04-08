import math

import numpy as np

from mc_average import mc_average


DEFAULT_ANALYSIS_SPEC = {
    "min_tail": 8,
    "first_window_fraction": 0.25,
    "last_window_fraction": 0.5,
    "stability_z": 2.0,
    "target_stderr": None,
    "target_stderr_by_observable": {},
    "target_observable": None,
}


def normalize_analysis_spec(analysis_spec=None):
    spec = dict(DEFAULT_ANALYSIS_SPEC)
    if analysis_spec is not None:
        spec.update(dict(analysis_spec))

    spec["min_tail"] = int(spec["min_tail"])
    spec["first_window_fraction"] = float(spec["first_window_fraction"])
    spec["last_window_fraction"] = float(spec["last_window_fraction"])
    spec["stability_z"] = float(spec["stability_z"])
    spec["target_stderr_by_observable"] = dict(spec.get("target_stderr_by_observable", {}))

    if spec["min_tail"] < 4:
        raise ValueError("analysis min_tail must be at least 4")
    if not 0.0 < spec["first_window_fraction"] < 1.0:
        raise ValueError("analysis first_window_fraction must lie in (0, 1)")
    if not 0.0 < spec["last_window_fraction"] < 1.0:
        raise ValueError("analysis last_window_fraction must lie in (0, 1)")
    if spec["stability_z"] <= 0.0:
        raise ValueError("analysis stability_z must be positive")
    if spec["target_stderr"] is not None and spec["target_stderr"] <= 0.0:
        raise ValueError("analysis target_stderr must be positive")
    for observable_name, target_stderr in spec["target_stderr_by_observable"].items():
        if target_stderr <= 0.0:
            raise ValueError(f"analysis target_stderr_by_observable[{observable_name!r}] must be positive")

    return spec


def _window_length(tail_length, fraction):
    return max(2, min(tail_length // 2, int(round(tail_length * fraction))))


def _combined_z_score(mean_a, err_a, mean_b, err_b):
    combined_err = math.hypot(err_a, err_b)
    if combined_err == 0.0:
        return 0.0 if math.isclose(mean_a, mean_b, rel_tol=0.0, abs_tol=1e-15) else math.inf
    return abs(mean_a - mean_b) / combined_err


def _burn_in_candidate_diagnostics(values, start_index, first_window_fraction, last_window_fraction):
    tail = values[start_index:]
    tail_length = tail.size
    first_length = _window_length(tail_length, first_window_fraction)
    last_length = _window_length(tail_length, last_window_fraction)

    if first_length + last_length > tail_length:
        first_length = max(2, tail_length // 3)
        last_length = max(2, tail_length // 3)

    first_window = tail[:first_length]
    last_window = tail[-last_length:]

    first_mean, first_stderr, _, _, _ = mc_average(first_window)
    last_mean, last_stderr, _, _, _ = mc_average(last_window)
    return {
        "first_mean": first_mean,
        "first_stderr": first_stderr,
        "last_mean": last_mean,
        "last_stderr": last_stderr,
        "z_score": _combined_z_score(first_mean, first_stderr, last_mean, last_stderr),
        "first_window_length": int(first_length),
        "last_window_length": int(last_length),
    }


def find_stable_start(values, analysis_spec=None):
    spec = normalize_analysis_spec(analysis_spec)
    x = np.asarray(values, dtype=float)
    n = x.size

    if n == 0:
        return {
            "start_index": 0,
            "stable": False,
            "z_score": math.inf,
            "reason": "empty_series",
            "first_window_length": 0,
            "last_window_length": 0,
        }

    if n < spec["min_tail"]:
        return {
            "start_index": 0,
            "stable": False,
            "z_score": math.inf,
            "reason": "too_short",
            "first_window_length": 0,
            "last_window_length": 0,
        }

    best_start = 0
    best_z = math.inf
    best_diagnostics = None

    max_start = n - spec["min_tail"]
    for start_index in range(max_start + 1):
        diagnostics = _burn_in_candidate_diagnostics(
            x,
            start_index,
            spec["first_window_fraction"],
            spec["last_window_fraction"],
        )
        z_score = diagnostics["z_score"]
        if z_score < best_z:
            best_start = start_index
            best_z = z_score
            best_diagnostics = diagnostics
        if z_score <= spec["stability_z"]:
            return {
                "start_index": int(start_index),
                "stable": True,
                "z_score": float(z_score),
                "reason": "threshold_met",
                "first_window_length": diagnostics["first_window_length"],
                "last_window_length": diagnostics["last_window_length"],
            }

    return {
        "start_index": int(best_start),
        "stable": False,
        "z_score": float(best_z),
        "reason": "best_available",
        "first_window_length": best_diagnostics["first_window_length"],
        "last_window_length": best_diagnostics["last_window_length"],
    }


def _recommend_additional_measurements(tail_length, stderr, target_stderr):
    if target_stderr is None or not math.isfinite(stderr):
        return None
    if stderr <= target_stderr:
        return {
            "target_stderr": float(target_stderr),
            "recommended_tail_measurements": int(tail_length),
            "recommended_additional_measurements": 0,
        }

    scale = (stderr / target_stderr) ** 2
    recommended_tail_measurements = max(tail_length + 1, int(math.ceil(tail_length * scale)))
    return {
        "target_stderr": float(target_stderr),
        "recommended_tail_measurements": int(recommended_tail_measurements),
        "recommended_additional_measurements": int(recommended_tail_measurements - tail_length),
    }


def analyze_time_series(values, measurement_stride=1, analysis_spec=None, observable_name=None):
    spec = normalize_analysis_spec(analysis_spec)
    x = np.asarray(values, dtype=float)
    n = x.size

    burn_in = find_stable_start(x, spec)
    start_index = burn_in["start_index"]
    tail = x[start_index:]

    mean, stderr, tau_int, n_eff, _ = mc_average(tail)

    target_stderr = spec["target_stderr_by_observable"].get(observable_name, spec["target_stderr"])
    recommendation = _recommend_additional_measurements(tail.size, stderr, target_stderr)

    result = {
        "series_length": int(n),
        "stable": bool(burn_in["stable"]),
        "burn_in_index": int(start_index),
        "burn_in_period": int(start_index * measurement_stride),
        "tail_length": int(tail.size),
        "mean": float(mean) if math.isfinite(mean) else None,
        "stderr": float(stderr) if math.isfinite(stderr) else None,
        "tau_int": float(tau_int) if math.isfinite(tau_int) else None,
        "n_eff": float(n_eff) if math.isfinite(n_eff) else None,
        "stability_z_score": float(burn_in["z_score"]) if math.isfinite(burn_in["z_score"]) else None,
        "stability_threshold": float(spec["stability_z"]),
        "stability_reason": burn_in["reason"],
        "first_window_length": int(burn_in["first_window_length"]),
        "last_window_length": int(burn_in["last_window_length"]),
    }

    if recommendation is not None:
        recommended_total_measurements = start_index + recommendation["recommended_tail_measurements"]
        result.update(
            {
                "target_stderr": recommendation["target_stderr"],
                "recommended_total_measurements": int(recommended_total_measurements),
                "recommended_additional_measurements": int(
                    max(0, recommended_total_measurements - n)
                ),
                "recommended_additional_periods": int(
                    max(0, recommended_total_measurements - n) * measurement_stride
                ),
            }
        )

    return result


def analyze_measurements(measurements, measurement_stride=1, analysis_spec=None):
    spec = normalize_analysis_spec(analysis_spec)
    observable_analysis = {
        observable_name: analyze_time_series(
            values,
            measurement_stride=measurement_stride,
            analysis_spec=spec,
            observable_name=observable_name,
        )
        for observable_name, values in measurements.items()
    }

    primary_observable = spec["target_observable"]
    if primary_observable is None and observable_analysis:
        primary_observable = next(iter(observable_analysis))

    result = {
        "series_definition": (
            "Each time-series point is the observer value at one measurement checkpoint "
            "after averaging over all walkers."
        ),
        "measurement_stride_periods": int(measurement_stride),
        "config": {
            "min_tail": int(spec["min_tail"]),
            "first_window_fraction": float(spec["first_window_fraction"]),
            "last_window_fraction": float(spec["last_window_fraction"]),
            "stability_z": float(spec["stability_z"]),
            "target_stderr": spec["target_stderr"],
            "target_stderr_by_observable": dict(spec["target_stderr_by_observable"]),
            "target_observable": primary_observable,
        },
        "observables": observable_analysis,
    }

    if primary_observable in observable_analysis:
        result["primary_observable"] = primary_observable
        result["primary_analysis"] = observable_analysis[primary_observable]

    return result
