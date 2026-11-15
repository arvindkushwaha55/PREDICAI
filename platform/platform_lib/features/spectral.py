"""Frequency-domain features for bearing vibration windows (wave 9).

Wave 4's leakage-safe CWRU re-evaluation and wave 5's CWRU<->Paderborn
transfers used ONLY the 10 time-domain statistics of
`platform_lib.adapters.cwru.window_features` — and fault *location*
(inner vs outer race) did not survive either honest protocol
(recording-wise InnerRace recall 0/177; cross-rig transfer accuracy
0.253 / 0.125 with Normal F1 = 0.0 in both directions). Bearing faults
are physically *spectral* phenomena (characteristic defect frequencies
and their harmonics/sidebands), so wave 9 adds a frequency-domain
feature set on top of — never instead of — the time-domain set, and
re-runs the two honest protocols unchanged.

SAMPLING-RATE COMPARABILITY (binding design choice): CWRU records at
12 kHz, Paderborn at 64 kHz, so the same 2048-sample window spans
170.7 ms vs 32.0 ms and any feature expressed in Hz or in absolute
amplitude is rig-specific by construction. Every feature here is
therefore expressed in RATE-FREE units:

- frequencies as fractions of Nyquist (0..1), never Hz;
- band edges as fractions of Nyquist, identical for both datasets;
- band energies normalised by the window's total spectral energy;
- envelope-spectrum peak amplitude normalised by the envelope
  spectrum's total magnitude (a concentration ratio, 0..1).

The price of rate-freedom is also stated plainly: a defect frequency
that sits at 0.02 x Nyquist on one rig and 0.10 x Nyquist on the other
(because shaft speeds and sampling rates differ independently) is NOT
aligned by this normalisation. These features remove the sampling-rate
confound only; they do not align shaft-speed-relative frequencies.

Feature list (14 spectral/envelope features; the combined set is the
10 time-domain features followed by these 14, in this order):

FFT magnitude features (rFFT of the demeaned window, DC bin excluded):
    dom_freq_frac         frequency of the largest magnitude bin,
                          as a fraction of Nyquist
    spectral_centroid_frac  magnitude-weighted mean frequency / Nyquist
    spectral_entropy      Shannon entropy of the normalised magnitude
                          spectrum, divided by log(n_bins) -> 0..1
    band_1 .. band_8      energy (squared magnitude) in 8 log-spaced
                          bands over (0.005 .. 1.0] x Nyquist, each
                          divided by total spectral energy; energy
                          below 0.005 x Nyquist is folded into band_1

Envelope features (Hilbert envelope of the window, demeaned):
    env_peak_amp_norm     largest envelope-spectrum magnitude divided
                          by the sum of envelope-spectrum magnitudes
    env_peak_freq_frac    frequency of that peak, fraction of Nyquist
    env_kurtosis          Fisher (excess) kurtosis of the envelope
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis

from ..adapters.cwru import FEATURE_COLUMNS as TIME_FEATURE_COLUMNS
from ..adapters.cwru import window_features

N_BANDS = 8
BAND_MIN_FRAC = 0.005  # lowest band edge, as a fraction of Nyquist

SPECTRAL_FEATURE_COLUMNS = (
    ["dom_freq_frac", "spectral_centroid_frac", "spectral_entropy"]
    + [f"band_{i}" for i in range(1, N_BANDS + 1)]
    + ["env_peak_amp_norm", "env_peak_freq_frac", "env_kurtosis"]
)

COMBINED_FEATURE_COLUMNS = TIME_FEATURE_COLUMNS + SPECTRAL_FEATURE_COLUMNS

# Band edges as fractions of Nyquist: log-spaced, identical at any
# sampling rate. BAND_EDGES[k] is the lower edge of band k+1.
BAND_EDGES = np.geomspace(BAND_MIN_FRAC, 1.0, N_BANDS + 1)


def _spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """rFFT magnitudes (DC excluded) and bin frequencies as fractions
    of Nyquist, for a real 1-D signal of length N (bin k <-> 2k/N)."""
    n = len(x)
    mag = np.abs(np.fft.rfft(x))[1:]
    freqs_frac = 2.0 * np.arange(1, len(mag) + 1) / n
    return mag, freqs_frac


def spectral_features(window: np.ndarray) -> dict:
    """The 14 frequency-domain features of one vibration window.

    Rate-free by construction (see module docstring): the sampling
    rate is never used, so the identical function applies to CWRU
    (12 kHz) and Paderborn (64 kHz) windows.
    """
    x = np.asarray(window, dtype=np.float64)
    x_dm = x - np.mean(x)

    mag, freqs = _spectrum(x_dm)
    total_mag = float(np.sum(mag))
    energy = mag ** 2
    total_energy = float(np.sum(energy))

    if total_mag > 0 and total_energy > 0:
        dom_freq_frac = float(freqs[int(np.argmax(mag))])
        centroid = float(np.sum(freqs * mag) / total_mag)
        p = mag / total_mag
        entropy = float(-np.sum(p * np.log(p + 1e-300))
                        / np.log(len(mag)))
        band_idx = np.clip(
            np.searchsorted(BAND_EDGES, freqs, side="right") - 1,
            0, N_BANDS - 1)
        band_energy = np.bincount(
            band_idx, weights=energy, minlength=N_BANDS) / total_energy
    else:  # silent/constant window: every spectral feature is 0
        dom_freq_frac, centroid, entropy = 0.0, 0.0, 0.0
        band_energy = np.zeros(N_BANDS)

    envelope = np.abs(hilbert(x_dm))
    env_kurt = float(kurtosis(envelope))
    if not np.isfinite(env_kurt):  # constant envelope (silent window)
        env_kurt = 0.0
    env_dm = envelope - np.mean(envelope)
    env_mag, env_freqs = _spectrum(env_dm)
    env_total = float(np.sum(env_mag))
    if env_total > 0:
        peak = int(np.argmax(env_mag))
        env_peak_amp_norm = float(env_mag[peak] / env_total)
        env_peak_freq_frac = float(env_freqs[peak])
    else:
        env_peak_amp_norm, env_peak_freq_frac = 0.0, 0.0

    feats = {
        "dom_freq_frac": dom_freq_frac,
        "spectral_centroid_frac": centroid,
        "spectral_entropy": entropy,
        "env_peak_amp_norm": env_peak_amp_norm,
        "env_peak_freq_frac": env_peak_freq_frac,
        "env_kurtosis": env_kurt,
    }
    for i in range(N_BANDS):
        feats[f"band_{i + 1}"] = float(band_energy[i])
    return {name: feats[name] for name in SPECTRAL_FEATURE_COLUMNS}


def combined_window_features(window: np.ndarray) -> dict:
    """The wave-9 combined set: the untouched 10 time-domain features
    of `cwru.window_features` plus the 14 spectral/envelope features
    of this module, keyed exactly by COMBINED_FEATURE_COLUMNS."""
    feats = window_features(window)
    feats.update(spectral_features(window))
    return {name: feats[name] for name in COMBINED_FEATURE_COLUMNS}
