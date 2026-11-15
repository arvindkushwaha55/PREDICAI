"""Order-domain (shaft-speed-relative) bearing features (wave 12).

Waves 4/5/9 established what does NOT align bearing-fault signatures
across rigs: the 10 time-domain statistics (recording-wise CWRU
0.665, transfers 0.253/0.125) and wave 9's rate-free spectral set
(0.679; transfers 0.210/0.187). The rate-free spectral features
removed the SAMPLING-rate confound but, by their own design note,
"do not align shaft-speed-relative defect frequencies": a defect
impact train repeats at a fixed multiple of SHAFT SPEED (its
characteristic order), and shaft speed varies independently of
sampling rate across the two rigs. This module is the documented
next attack: features read off the envelope spectrum at the
characteristic ORDERS of the actual bearing in the rig, so the same
physical defect lands on the same feature on both rigs by
construction.

PHYSICS — characteristic orders (multiples of shaft speed), from the
standard kinematic formulas with contact angle theta, ball count n,
ball diameter Bd and pitch diameter Pd:

    r    = (Bd / Pd) * cos(theta)
    BPFO = (n/2) * (1 - r)          ball pass frequency, outer race
    BPFI = (n/2) * (1 + r)          ball pass frequency, inner race
    FTF  = (1/2) * (1 - r)          fundamental train (cage) frequency
    BSF  = (Pd / (2*Bd)) * (1 - r^2)  ball SPIN frequency

A defect ON a ball strikes the inner and outer raceway once each per
ball revolution, so the ball-defect repetition rate seen in an
envelope spectrum is 2*BSF, not BSF. The `bsf` feature below is
placed at 2*BSF (the ball-DEFECT order) — this is also the value the
CWRU Data Center publishes as its "rolling element" multiplier
(4.7135 for the 6205), and the record states the convention wherever
the number is quoted.

BEARING GEOMETRY (inputs, with sources):

- CWRU drive end — SKF 6205-2RS JEM (the test bearing of the CWRU
  Bearing Data Center): n = 9, Bd = 0.3126 in, Pd = 1.537 in,
  theta = 0 deg. The Data Center's own published multipliers are
  BPFI 5.4152 / BPFO 3.5848 / rolling-element 4.7135 / FTF 0.39828;
  the formulas above reproduce them from the geometry to <= 8e-5
  (BPFO 3.58477, BPFI 5.41523, 2*BSF 4.71342, FTF 0.39831), so the
  computed values are used and the published ones kept alongside as
  the cross-check.
- Paderborn — 6203 (the test bearing of the KAt-DataCenter):
  n = 8, Bd = 6.75 mm, Pd = 28.55 mm, theta = 0 deg. Computed:
  BPFO 3.05429, BPFI 4.94571, 2*BSF 3.99322, FTF 0.38179 — matching
  the values computed from the same nominal geometry in the
  Paderborn literature (BPFI 4.9457 / BPFO 3.0543 / BSF 1.9966).

SHAFT SPEEDS (per recording/measurement, supplied by the caller):
- CWRU: 10 of the 12 wave-4 recordings carry their own X<id>RPM
  scalar in the .mat file (all 0 hp recordings read 1796-1797 rpm).
  The two that do not (98.mat / 99.mat, the 1 hp / 2 hp normal
  baselines) take the CWRU Data Center catalogue's nominal speed
  for their load: 1772 rpm (1 hp) / 1750 rpm (2 hp); the catalogue's
  0 hp nominal is 1797 rpm, consistent with the in-file scalars.
- Paderborn: the condition code in every measurement filename
  encodes the nominal speed (N15 = 1500 rpm, N09 = 900 rpm;
  platform_lib.adapters.paderborn.parse_condition). Cross-checked
  against the synchronously recorded `speed` channel in the files
  themselves: N09 file mean 899.7 rpm, N15 file mean 1499.6 rpm
  (std 0.18 rpm) — the nominal values are used, a <= 0.03% error,
  far inside the +-2% feature bands.

FEATURE DEFINITION (per 2048-sample window; 9 features):
The Hilbert envelope of the demeaned window is demeaned and its
magnitude spectrum computed with 8x zero-padding (nfft = 16384;
padding interpolates bin placement, it does NOT improve true
resolution). Each bin's frequency is expressed in orders
(Hz / shaft Hz). For each characteristic order o in
{BPFO, BPFI, BSF(defect), FTF} and its 2nd harmonic 2o, the feature
is the share of total envelope-spectrum energy inside the
+-2%-of-target band around the order:

    share(o) = sum(|E|^2 over bins with order in o*[0.98, 1.02])
               / sum(|E|^2 over all non-DC bins)

If no bin centre falls inside a band (possible at FTF, whose band
is narrower than the bin spacing on the short Paderborn windows),
the single nearest bin's energy is used — the fallback is part of
the definition, not an afterthought. The ninth feature,
`order_entropy`, is the Shannon entropy of the envelope magnitude
spectrum restricted to orders in (0, 12], normalised by log(n_bins)
to 0..1 (a spread measure over the same order axis).

RESOLUTION REALITY (stated, not hidden): zero-padding interpolates;
the TRUE order resolution of a window is 1/(revolutions per window).
A CWRU window spans 5.11 shaft revolutions (delta-order 0.196); a
Paderborn window spans 0.80 revolutions at N15 and 0.48 at N09
(delta-order 1.25 / 2.08) — coarser than the spacing between the
characteristic orders themselves. The order features on Paderborn
windows are therefore blunt instruments by construction of the
2048-sample protocol the waves share; the wave-12 records say so.

RESIDUAL CONFOUNDS (what order alignment does NOT fix): load- and
slip-dependent deviation of true defect rates from the kinematic
orders (typically ~1-2%, the reason for the +-2% bands); speed
estimation error (nominal vs instantaneous speed); different sensor
placement, mounting and rig transfer functions; different fault
geometries (EDM single-point defects of different sizes, Paderborn
KA01/KI01 artificial vs CWRU EDM). Per-dataset orders remove the
geometry difference between the 6205 and the 6203; nothing here
removes the rigs themselves.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert

from ..adapters.cwru import FEATURE_COLUMNS as TIME_FEATURE_COLUMNS
from ..adapters.cwru import window_features

BAND_TOL = 0.02        # +-2% of the target order
PAD_FACTOR = 8         # zero-padding for bin placement (interpolation)
MAX_ORDER_ENTROPY = 12.0

ORDER_FEATURE_COLUMNS = [
    "ord_bpfo", "ord_bpfi", "ord_bsf", "ord_ftf",
    "ord_bpfo_h2", "ord_bpfi_h2", "ord_bsf_h2", "ord_ftf_h2",
    "order_entropy",
]

ORDER_COMBINED_FEATURE_COLUMNS = (
    TIME_FEATURE_COLUMNS + ORDER_FEATURE_COLUMNS)


@dataclass(frozen=True)
class BearingSpec:
    """Bearing geometry + provenance for characteristic-order maths."""

    name: str
    n_balls: int
    ball_diameter: float     # same unit as pitch_diameter (ratio only)
    pitch_diameter: float
    contact_angle_deg: float
    geometry_source: str
    published_multipliers: dict   # cross-check values, as published

    def fault_orders(self) -> dict:
        """Characteristic orders from the kinematic formulas.

        Returns bpfo / bpfi / ftf, the ball SPIN frequency
        `bsf_spin`, and `bsf` = 2*bsf_spin — the ball-DEFECT order
        at which a ball fault repeats in an envelope spectrum (and
        the multiplier CWRU publishes for the rolling element).
        """
        r = (self.ball_diameter / self.pitch_diameter) * np.cos(
            np.deg2rad(self.contact_angle_deg))
        bpfo = (self.n_balls / 2.0) * (1.0 - r)
        bpfi = (self.n_balls / 2.0) * (1.0 + r)
        ftf = 0.5 * (1.0 - r)
        bsf_spin = (self.pitch_diameter / (2.0 * self.ball_diameter)) \
            * (1.0 - r ** 2)
        return {"bpfo": float(bpfo), "bpfi": float(bpfi),
                "ftf": float(ftf), "bsf_spin": float(bsf_spin),
                "bsf": float(2.0 * bsf_spin)}

    def documentation(self) -> dict:
        orders = self.fault_orders()
        return {
            "bearing": self.name,
            "n_balls": self.n_balls,
            "ball_diameter": self.ball_diameter,
            "pitch_diameter": self.pitch_diameter,
            "contact_angle_deg": self.contact_angle_deg,
            "geometry_source": self.geometry_source,
            "computed_orders": {k: round(v, 5)
                                for k, v in orders.items()},
            "published_multipliers_crosscheck":
                self.published_multipliers,
            "bsf_convention": "the `bsf` feature sits at 2*BSF "
                              "(ball-defect repetition rate: a ball "
                              "defect strikes both raceways once per "
                              "ball revolution); `bsf_spin` is the "
                              "pure ball spin frequency",
        }


CWRU_6205 = BearingSpec(
    name="SKF 6205-2RS JEM (CWRU drive end)",
    n_balls=9, ball_diameter=0.3126, pitch_diameter=1.537,
    contact_angle_deg=0.0,
    geometry_source="CWRU Bearing Data Center bearing information "
                    "(drive-end test bearing SKF 6205-2RS JEM): "
                    "9 balls, ball diameter 0.3126 in, pitch "
                    "diameter 1.537 in, contact angle 0 deg",
    published_multipliers={"bpfi": 5.4152, "bpfo": 3.5848,
                           "bsf_defect_2xbsf": 4.7135, "ftf": 0.39828},
)

PADERBORN_6203 = BearingSpec(
    name="6203 (Paderborn KAt-DataCenter test bearing)",
    n_balls=8, ball_diameter=6.75, pitch_diameter=28.55,
    contact_angle_deg=0.0,
    geometry_source="6203 nominal geometry as used for the "
                    "Paderborn KAt-DataCenter bearing: 8 balls, "
                    "ball diameter 6.75 mm, pitch diameter "
                    "28.55 mm, contact angle 0 deg",
    published_multipliers={"bpfi": 4.9457, "bpfo": 3.0543,
                           "bsf_spin": 1.9966},
)


def _envelope_spectrum(window: np.ndarray, fs_hz: float,
                       shaft_rpm: float) -> tuple[np.ndarray, np.ndarray]:
    """Envelope magnitude spectrum and its order axis.

    Returns (magnitudes, orders) for rFFT bins of the demeaned
    Hilbert envelope, zero-padded to PAD_FACTOR * len(window).
    Bin 0 (DC) is included in the arrays; callers exclude it.
    """
    x = np.asarray(window, dtype=np.float64)
    x = x - np.mean(x)
    env = np.abs(hilbert(x))
    env = env - np.mean(env)
    nfft = PAD_FACTOR * len(x)
    mag = np.abs(np.fft.rfft(env, n=nfft))
    shaft_hz = shaft_rpm / 60.0
    orders = (np.arange(len(mag)) * (fs_hz / nfft)) / shaft_hz
    return mag, orders


def _band_share(mag: np.ndarray, orders: np.ndarray, target: float,
                total_energy: float) -> float:
    """Share of envelope-spectrum energy within +-BAND_TOL of an
    order; nearest single bin if the band holds no bin centre."""
    if total_energy <= 0:
        return 0.0
    in_band = np.abs(orders - target) <= BAND_TOL * target
    if not in_band.any():
        in_band = np.zeros(len(mag), dtype=bool)
        in_band[int(np.argmin(np.abs(orders - target)))] = True
    return float(np.sum(mag[in_band] ** 2) / total_energy)


def order_features(window: np.ndarray, fs_hz: float, shaft_rpm: float,
                   spec: BearingSpec) -> dict:
    """The 9 order-domain features of one vibration window.

    Shaft-speed-relative by construction: the characteristic orders
    are properties of the bearing geometry alone, so the same
    physical defect feeds the same feature at any speed, sampling
    rate or (via `spec`) bearing type. See the module docstring for
    the band definition, the nearest-bin fallback, the resolution
    limits and the residual confounds.
    """
    mag, orders = _envelope_spectrum(window, fs_hz, shaft_rpm)
    total_energy = float(np.sum(mag[1:] ** 2))
    o = spec.fault_orders()
    feats = {}
    for key in ("bpfo", "bpfi", "bsf", "ftf"):
        feats[f"ord_{key}"] = _band_share(mag, orders, o[key],
                                          total_energy)
        feats[f"ord_{key}_h2"] = _band_share(mag, orders, 2.0 * o[key],
                                             total_energy)
    in_range = (orders > 0) & (orders <= MAX_ORDER_ENTROPY)
    m = mag[in_range]
    total_mag = float(np.sum(m))
    if len(m) >= 2 and total_mag > 0:
        p = m / total_mag
        feats["order_entropy"] = float(
            -np.sum(p * np.log(p + 1e-300)) / np.log(len(m)))
    else:
        feats["order_entropy"] = 0.0
    return {name: feats[name] for name in ORDER_FEATURE_COLUMNS}


def order_window_features(window: np.ndarray, fs_hz: float,
                          shaft_rpm: float,
                          spec: BearingSpec) -> dict:
    """The wave-12 combined set: the untouched 10 time-domain
    features of `cwru.window_features` plus the 9 order-domain
    features of this module, keyed by ORDER_COMBINED_FEATURE_COLUMNS."""
    feats = window_features(window)
    feats.update(order_features(window, fs_hz, shaft_rpm, spec))
    return {name: feats[name] for name in ORDER_COMBINED_FEATURE_COLUMNS}
