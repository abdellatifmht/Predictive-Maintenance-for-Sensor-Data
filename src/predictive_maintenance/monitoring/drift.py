from dataclasses import dataclass

import numpy as np
from scipy import stats

PSI_THRESHOLD_ALERT = 0.25
PSI_THRESHOLD_WARNING = 0.10
KS_PVALUE_THRESHOLD = 0.05


@dataclass
class DriftResult:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float

    @property
    def psi_level(self) -> str:
        if self.psi >= PSI_THRESHOLD_ALERT:
            return "alert"
        elif self.psi >= PSI_THRESHOLD_WARNING:
            return "warning"
        return "ok"

    @property
    def ks_flagged(self) -> bool:
        return self.ks_pvalue < KS_PVALUE_THRESHOLD

    @property
    def is_drifted(self) -> bool:
        return self.psi_level == "alert" or self.ks_flagged


def _psi_bins(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(reference, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf
    return np.unique(edges)


def population_stability_index(reference: np.ndarray, current: np.ndarray, n_bins: int = 10, epsilon: float = 1e-4) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    edges = _psi_bins(reference, n_bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(len(reference), 1) + epsilon
    cur_pct = cur_counts / max(len(current), 1) + epsilon

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compare_distributions(feature: str, reference: np.ndarray, current: np.ndarray) -> DriftResult:
    psi = population_stability_index(reference, current)
    ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
    return DriftResult(feature=feature, psi=psi, ks_statistic=float(ks_stat), ks_pvalue=float(ks_pvalue))


def compare_all_features(reference_df, current_df, features: list[str]) -> list[DriftResult]:
    results = []
    for feature in features:
        if feature not in current_df.columns or current_df[feature].dropna().empty:
            continue
        result = compare_distributions(
            feature,
            reference_df[feature].dropna().values,
            current_df[feature].dropna().values,
        )
        results.append(result)
    return results
