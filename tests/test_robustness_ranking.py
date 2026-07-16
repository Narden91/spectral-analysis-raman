from __future__ import annotations

import pandas as pd

from src.analysis.robustness_ranking import compute_evidence_flags


def test_compute_evidence_flags_is_deterministic() -> None:
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

    ranking = compute_evidence_flags(
        contrasts,
        consistency,
        effect_size_threshold=0.147,
        perm_p_threshold=0.10,
        anchor_feature="oxidative_index",
    )

    assert ranking["feature_name"].tolist() == [
        "membrane_integrity_index",
        "oxidative_index",
        "protein_remodeling_index",
    ]
    anchor = ranking.loc[ranking["feature_name"] == "oxidative_index"].iloc[0]
    assert bool(anchor["effect_size_flag"])
    assert bool(anchor["bootstrap_ci_excludes_zero"])
    assert bool(anchor["permutation_flag"])
    assert bool(anchor["is_anchor_feature"])
    assert "robust_score" not in ranking.columns
    assert "evidence_tier" not in ranking.columns
