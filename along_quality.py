"""Along-axis measurement quality models for SDSLO single-line tracking.

The particle filter historically used a fixed along-position likelihood width.
For SDSLO real captures the along/vertical match can become ambiguous, so this
module keeps the calibration rule explicit and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-9


def _tag_float(x: float) -> str:
    s = f"{float(x):g}"
    return s.replace("-", "m").replace(".", "p")


def normalize_subject_quality(qv, q_p10: float, q_p90: float) -> np.ndarray:
    """Subject-normalize qv using capture percentiles, safely handling flats."""
    q = np.asarray(qv, dtype=np.float64)
    out_shape = q.shape
    q = q.reshape(-1)
    p10 = float(q_p10)
    p90 = float(q_p90)
    den = p90 - p10
    if not np.isfinite(p10) or not np.isfinite(p90) or den <= EPS:
        out = np.full(q.shape, 0.5, dtype=np.float64)
    else:
        out = (q - p10) / den
        out = np.where(np.isfinite(out), out, 0.0)
        out = np.clip(out, 0.0, 1.0)
    return out.reshape(out_shape)


@dataclass(frozen=True)
class AlongQualityModel:
    """Map per-line quality to an effective along measurement sigma.

    ``constant`` reproduces the historical PF behavior exactly:
    ``sigma_along_eff = sigma_along``.

    ``qv_power`` uses subject-normalized qv:
    ``sigma = sigma_min + (sigma_max - sigma_min) * (1 - q_norm)**gamma``.
    """

    kind: str = "constant"
    sigma_min: float = 2.0
    sigma_max: float = 2.0
    gamma: float = 1.0
    q_p10: float = 0.0
    q_p90: float = 1.0
    source: str = "qv"

    def __post_init__(self) -> None:
        kind = self.kind.replace("-", "_")
        object.__setattr__(self, "kind", kind)
        if kind not in {"constant", "qv_power"}:
            raise ValueError(f"unknown along-quality model {self.kind!r}")
        if self.sigma_min <= 0 or self.sigma_max <= 0:
            raise ValueError("sigma_min and sigma_max must be positive")
        if self.sigma_max < self.sigma_min:
            raise ValueError("sigma_max must be >= sigma_min")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")

    @classmethod
    def constant(cls) -> "AlongQualityModel":
        return cls(kind="constant")

    @classmethod
    def qv_power(cls, sigma_min: float, sigma_max: float, gamma: float,
                 q_p10: float, q_p90: float) -> "AlongQualityModel":
        return cls(
            kind="qv_power",
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            gamma=float(gamma),
            q_p10=float(q_p10),
            q_p90=float(q_p90),
            source="qv",
        )

    @classmethod
    def fit_qv_power(cls, qv, sigma_min: float, sigma_max: float,
                     gamma: float) -> "AlongQualityModel":
        q = np.asarray(qv, dtype=np.float64)
        q = q[np.isfinite(q)]
        if q.size:
            p10, p90 = np.percentile(q, [10.0, 90.0])
        else:
            p10, p90 = 0.0, 1.0
        return cls.qv_power(sigma_min, sigma_max, gamma, float(p10), float(p90))

    def q_norm(self, qv) -> np.ndarray:
        return normalize_subject_quality(qv, self.q_p10, self.q_p90)

    def sigma(self, qv, base_sigma: float) -> np.ndarray:
        q = np.asarray(qv, dtype=np.float64)
        if self.kind == "constant":
            return np.full(q.shape, float(base_sigma), dtype=np.float64)
        qn = self.q_norm(q)
        sig = self.sigma_min + (self.sigma_max - self.sigma_min) * (1.0 - qn) ** self.gamma
        return np.asarray(sig, dtype=np.float64)

    def sigma_scalar(self, qv: float, base_sigma: float) -> float:
        return float(self.sigma(np.asarray(qv, dtype=np.float64), base_sigma).reshape(-1)[0])

    def config_tag(self) -> str:
        if self.kind == "constant":
            return "constant"
        return (
            f"qv_power_s{_tag_float(self.sigma_min)}"
            f"_{_tag_float(self.sigma_max)}"
            f"_g{_tag_float(self.gamma)}"
        )

    def as_dict(self) -> dict[str, float | str]:
        return dict(
            kind=self.kind,
            sigma_min=float(self.sigma_min),
            sigma_max=float(self.sigma_max),
            gamma=float(self.gamma),
            q_p10=float(self.q_p10),
            q_p90=float(self.q_p90),
            source=self.source,
        )
