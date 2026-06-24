from __future__ import annotations

import pandas as pd

from src.analysis.endpoint_consistency import compute_endpoint_consistency


def test_compute_endpoint_consistency_flags_same_sign_features() -> None:
    associations = pd.DataFrame(
        {
            "feature_name": ["oxidative_index", "oxidative_index", "oxidative_index", "membrane_integrity_index", "membrane_integrity_index"],
            "endpoint": ["moca_baseline", "moca_3m", "delta_moca", "moca_baseline", "delta_moca"],
            "partial_spearman_rho": [-0.4, -0.3, 0.1, 0.5, -0.2],
        }
    )

    result = compute_endpoint_consistency(
        associations,
        endpoints=["moca_baseline", "moca_3m", "delta_moca"],
        min_endpoint_agreement=2,
    )

    oxidative = result.loc[result["feature_name"] == "oxidative_index"].iloc[0]
    membrane = result.loc[result["feature_name"] == "membrane_integrity_index"].iloc[0]

    assert bool(oxidative["consistent"])
    assert oxidative["dominant_sign"] == "negative"
    assert oxidative["n_same_sign_endpoints"] == 2
    assert not bool(membrane["consistent"])
