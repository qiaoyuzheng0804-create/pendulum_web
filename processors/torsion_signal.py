# -*- coding: utf-8 -*-
"""Cycle diagnostics without assuming that the equilibrium angle is zero."""
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import detrend, find_peaks


def finite_runs(t, y):
    """Never join extrema or spectra across missing frames or timestamp gaps."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if t.ndim != 1 or y.shape != t.shape:
        raise ValueError("Time and angle arrays must be one-dimensional and equally sized")
    if not np.isfinite(t).all() or (len(t) > 1 and np.any(np.diff(t) <= 0)):
        raise ValueError("Time must be finite and strictly increasing")
    if len(t) < 2:
        return []
    dt = float(np.median(np.diff(t)))
    valid = np.isfinite(y)
    breaks = np.r_[True, np.diff(t) > 1.8 * dt]
    starts = np.flatnonzero(valid & (breaks | ~np.r_[False, valid[:-1]]))
    ends = np.flatnonzero(valid & (np.r_[breaks[1:], True] | ~np.r_[valid[1:], False])) + 1
    return list(zip(starts, ends))


def estimate_period_fft(t, angles, min_cycles=3):
    t, y = np.asarray(t, float), np.asarray(angles, float)
    runs = finite_runs(t, y)
    if not runs:
        return None, None, 0.0
    start, end = max(runs, key=lambda pair: pair[1] - pair[0])
    if end - start < 32:
        return None, None, 0.0
    grid = t[start:end]
    # Interpolation here is exclusively a spectral diagnostic within a finite run.
    uniform = np.linspace(grid[0], grid[-1], len(grid))
    signal = detrend(np.interp(uniform, grid, y[start:end]), type="linear")
    if np.max(np.abs(signal)) < 1e-10:
        return None, None, 0.0
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    frequencies = np.fft.rfftfreq(len(signal), float(uniform[1] - uniform[0]))
    spectrum[0] = 0
    k = int(np.argmax(spectrum))
    if k == 0:
        return None, None, 0.0
    period = 1.0 / float(frequencies[k])
    return period, float(frequencies[k]), float((grid[-1] - grid[0]) / period)


def find_cycle_peaks(t, angles, rel_prom=0.25, min_prom_deg=0.03):
    """Detect maxima relative to local peak-to-peak range, including negative maxima."""
    if rel_prom < 0 or min_prom_deg <= 0:
        raise ValueError("Peak prominence thresholds must be nonnegative/positive")
    t, y = np.asarray(t, float), np.asarray(angles, float)
    peaks, prominences = [], []
    for start, end in finite_runs(t, y):
        if end - start < 8:
            continue
        segment = y[start:end]
        period, _, ncycles = estimate_period_fft(t[start:end], segment)
        if period is None or ncycles < 3:
            continue
        dt = float(np.median(np.diff(t[start:end])))
        samples = max(3, int(round(period / dt)))
        distance = max(1, int(0.55 * samples))
        window = min(len(segment), 2 * samples + 1)
        if window % 2 == 0:
            window -= 1
        span = (maximum_filter1d(segment, size=window, mode="nearest") -
                minimum_filter1d(segment, size=window, mode="nearest"))
        threshold = np.maximum(min_prom_deg, rel_prom * span)
        indices, props = find_peaks(segment, distance=distance,
                                    prominence=threshold, wlen=window)
        peaks.extend((indices + start).tolist())
        prominences.extend(props["prominences"].tolist())
    return np.asarray(peaks, dtype=int), np.asarray(prominences, dtype=float)


def cycle_envelope(t, angles, peaks):
    """Bracket each trough by two maxima and interpolate the upper value to its time."""
    t, y, peaks = np.asarray(t, float), np.asarray(angles, float), np.asarray(peaks, int)
    names = ("amp_t", "amp", "center", "upper", "lower", "period_s",
             "peak_left_index", "trough_index", "peak_right_index")
    result = {name: [] for name in names}
    if len(t) < 2 or len(peaks) < 2:
        return {name: np.asarray(values) for name, values in result.items()}
    dt = float(np.median(np.diff(t)))
    expected = float(np.median(np.diff(t[peaks])))
    for left, right in zip(peaks[:-1], peaks[1:]):
        if right - left < 2 or not np.isfinite(y[left:right + 1]).all():
            continue
        if np.any(np.diff(t[left:right + 1]) > 1.8 * dt) or t[right] - t[left] > 1.8 * expected:
            continue
        trough = int(left + np.argmin(y[left:right + 1]))
        if trough == left or trough == right:
            continue
        upper = float(np.interp(t[trough], t[[left, right]], y[[left, right]]))
        lower = float(y[trough])
        amplitude = (upper - lower) / 2
        if amplitude <= 0:
            continue
        values = (t[trough], amplitude, (upper + lower) / 2, upper, lower,
                  t[right] - t[left], left, trough, right)
        for name, value in zip(names, values):
            result[name].append(value)
    return {name: np.asarray(values, dtype=int if name.endswith("_index") else float)
            for name, values in result.items()}


def cycle_amplitudes(t, angles, peaks):
    envelope = cycle_envelope(t, angles, peaks)
    return envelope["amp_t"], envelope["amp"]


def decay_report(t, angles, peaks, n_blocks=4, min_pts=6):
    """Descriptive log-amplitude slopes; they do not identify a damping mechanism."""
    result = cycle_envelope(t, angles, peaks)
    result["segments"] = []
    tt, aa = result["amp_t"], result["amp"]
    if len(tt) < min_pts:
        return result
    for indices in np.array_split(np.arange(len(tt)), n_blocks):
        if len(indices) < min_pts:
            continue
        ts, amplitudes = tt[indices], aa[indices]
        slope, intercept = np.polyfit(ts, np.log(amplitudes), 1)
        beta = -float(slope)  # Preserve a negative slope estimate for growing amplitudes.
        period = float(np.median(result["period_s"][indices]))
        result["segments"].append(dict(
            t_mid=float(np.median(ts)),
            t_start=float(ts[0]),
            t_end=float(ts[-1]),
            beta=beta,
            beta_per_s=beta,
            decay_per_cycle_pct=float(100 * (-np.expm1(-beta * period))),
            n=int(len(indices)),
            n_peaks=int(len(indices)),
            amp_mid=float(np.median(amplitudes)),
        ))
    return result


def decay_rate(t, angles, peaks):
    tt, aa = cycle_amplitudes(t, angles, peaks)
    if len(tt) < 5:
        return None
    return -float(np.polyfit(tt, np.log(aa), 1)[0])
