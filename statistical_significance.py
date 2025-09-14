from math import sqrt
from typing import NamedTuple
from scipy.stats import norm

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
