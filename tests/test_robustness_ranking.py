from __future__ import annotations

import pandas as pd

from src.analysis.robustness_ranking import compute_robustness_scores


def test_compute_robustness_scores_is_deterministic() -> None:
    contrasts = pd.DataFrame(
        {
            "feature_name": ["oxidative_index", "membrane_integrity_index", "protein_remodeling_index"],
            "feature_family": ["named_index", "named_index", "named_index"],
            "status": ["ok", "ok", "ok"],
            "cliffs_delta": [0.8, 0.3, 0.1],
            "bootstrap_ci_lower": [0.2, -0.1, -0.2],
            "bootstrap_ci_upper": [0.9, 0.5, 0.3],
            "perm_p_value": [0.04, 0.2, 0.9],
        }
    )
    consistency = pd.DataFrame(
        {
            "feature_name": ["oxidative_index", "membrane_integrity_index"],
            "consistent": [True, True],
        }
    )

    ranking = compute_robustness_scores(
        contrasts,
        consistency,
        effect_size_threshold=0.147,
        perm_p_threshold=0.10,
        anchor_feature="oxidative_index",
    )

    assert ranking["feature_name"].tolist() == [
        "oxidative_index",
        "membrane_integrity_index",
        "protein_remodeling_index",
    ]
    assert ranking.loc[0, "robust_score"] == 4
    assert ranking.loc[0, "evidence_tier"] == "strong_trend"
    assert bool(ranking.loc[0, "is_anchor_feature"])
    assert ranking["rank"].tolist() == [1, 2, 3]
