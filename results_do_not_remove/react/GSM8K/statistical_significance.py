from math import sqrt
from typing import NamedTuple
from scipy.stats import norm, fisher_exact


class AccuracyComparisonResult(NamedTuple):
    z_score: float
    p_value: float
    is_significant: bool
    min_required_delta: float

def compare_accuracies(p1: float, p2: float, n: int, alpha: float = 0.05) -> AccuracyComparisonResult:
    """
    Vergleicht zwei Modellgenauigkeiten (p1, p2) mit je n Stichproben.
    
    Parameter:
        p1, p2: Genauigkeiten der Modelle (zwischen 0 und 1)
        n: Stichprobengröße (Anzahl getesteter Beispiele pro Modell)
        alpha: Signifikanzniveau (z.B. 0.05 für 95%)

    Rückgabe:
        AccuracyComparisonResult NamedTuple mit:
            - z_score
            - p_value (zweiseitig)
            - is_significant (True/False)
            - min_required_delta (Minimaler Unterschied, der für Signifikanz nötig wäre)
    """
    se1 = sqrt(p1 * (1 - p1) / n)
    se2 = sqrt(p2 * (1 - p2) / n)
    se_diff = sqrt(se1**2 + se2**2)
    
    delta = abs(p2 - p1)
    z = delta / se_diff if se_diff > 0 else 0.0
    p_value = 2 * (1 - norm.cdf(z))
    
    z_crit = norm.ppf(1 - alpha / 2)
    min_diff = z_crit * se_diff
    is_significant = z >= z_crit

    return AccuracyComparisonResult(
        z_score=z,
        p_value=p_value,
        is_significant=is_significant,
        min_required_delta=min_diff
    )

def compare_accuracies_v2(p1: float, n1: int, p2: float, n2: int, alpha: float = 0.05) -> AccuracyComparisonResult:
    """
    Compares two accuracies (p1, p2) evaluated on different sample sizes (n1, n2).
    
    Parameters:
        p1, p2: Accuracy of the models (between 0 and 1)
        n1, n2: Number of samples used to calculate p1 and p2
        alpha: Significance level (e.g., 0.05 for 95% confidence)

    Returns:
        AccuracyComparisonResult:
            - z_score
            - p_value (two-tailed)
            - is_significant (True/False)
            - min_required_delta (Minimum difference needed for significance)
    """
    se1 = sqrt(p1 * (1 - p1) / n1)
    se2 = sqrt(p2 * (1 - p2) / n2)
    se_diff = sqrt(se1**2 + se2**2)

    delta = abs(p2 - p1)
    z = delta / se_diff if se_diff > 0 else 0.0
    p_value = 2 * (1 - norm.cdf(z))

    z_crit = norm.ppf(1 - alpha / 2)
    min_diff = z_crit * se_diff
    is_significant = z >= z_crit

    return AccuracyComparisonResult(
        z_score=z,
        p_value=p_value,
        is_significant=is_significant,
        min_required_delta=min_diff
    )


def compare_accuracies_v3(p1: float, n1: int, p2: float, n2: int, alpha: float = 0.05) -> AccuracyComparisonResult:
    """
    Compares two accuracies. Uses z-test for large sample sizes (n1, n2 ≥ 30), otherwise Fisher’s exact test.
    
    Returns a consistent AccuracyComparisonResult with a dummy z-score and min_required_delta if Fisher is used.
    """
    # Use z-test if both sample sizes are large enough
    if min(n1, n2) >= 30:
        se1 = sqrt(p1 * (1 - p1) / n1)
        se2 = sqrt(p2 * (1 - p2) / n2)
        se_diff = sqrt(se1**2 + se2**2)

        delta = abs(p1 - p2)
        z = delta / se_diff if se_diff > 0 else 0.0
        p_value = 2 * (1 - norm.cdf(z))
        z_crit = norm.ppf(1 - alpha / 2)
        min_diff = z_crit * se_diff
        is_significant = z >= z_crit

        return AccuracyComparisonResult(
            z_score=z,
            p_value=p_value,
            is_significant=is_significant,
            min_required_delta=min_diff
        )
    else:
        # Use Fisher's exact test
        correct1 = round(p1 * n1)
        correct2 = round(p2 * n2)
        table = [[correct1, n1 - correct1], [correct2, n2 - correct2]]
        _, p_value = fisher_exact(table, alternative='two-sided')
        is_significant = p_value < alpha

        # No meaningful z-score or delta in Fisher case
        return AccuracyComparisonResult(
            z_score=float('nan'),
            p_value=p_value,
            is_significant=is_significant,
            min_required_delta=float('nan')
        )
