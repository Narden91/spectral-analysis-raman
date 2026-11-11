"""Step 4: Sensitivity analyses, secondary laboratory endpoints, and anchor associations.

Produces:
- ``covariate_and_subset_sensitivity.csv``
- ``outer_partition_seed_sensitivity.csv``
- ``secondary_endpoints_summary.csv``
- ``anchor_associations_continuous.csv``
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, get_args

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript_ml import (
    add_cohort_stage_args,
    ensure_output_dir,
    load_manuscript_config,
    resolve_experiment_dataset,
    write_run_provenance,
)
from src.analysis.io_utils import write_csv_atomic
from src.analysis.paired_lipid.cohort import PairedLipidConfig, SubsetMode
from src.analysis.paired_lipid.nested_cv import (
    evaluate_nested_cohort,
    summarize_anchor_associations,
)

COVARIATE_PRESETS: tuple[str, ...] = (
    "primary",
    "no_label_trend",
    "quadratic_label_trend",
    "quadratic_age_and_label",
    "pre_event_risk_factors",
)
SUBSET_MODES: tuple[SubsetMode, ...] = get_args(SubsetMode)


def _paired_metrics(res: dict[str, Any], *, with_mae: bool = False) -> dict[str, float]:
    """Paired-model summary columns shared by every sensitivity table, in output column order."""
    m_pair = res["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]
    row = {
        "delta_mse_paired": m_pair["delta_mse"],
        "relative_mse_reduction": m_pair["relative_mse_reduction"],
        "rmse_paired": m_pair["rmse_model"],
    }
    if with_mae:
        row["mae_paired"] = m_pair["mae_model"]
    row["delta_vs_strongest_comparator"] = res["paired_vs_strongest_comparator"]["delta_mse"]
    return row


def run_sensitivity_and_secondary(
    *,
    use_synthetic: bool = True,
    approve_cohort_evaluation: bool = False,
    clinical_master_path: str | Path = PROJECT_ROOT / "data" / "clinical_master.csv",
    output_dir: str | Path | None = None,
    compact_grid: bool = False,
    config_path: str | Path | None = None,
) -> dict[str, Path]:
    """Run all prespecified sensitivity analyses, secondary endpoints, and anchor summaries."""
    target_dir = ensure_output_dir(output_dir)
    base_cfg, raw_cfg = load_manuscript_config(config_path)
    anchor_cfg = raw_cfg["anchors"]
    eval_memo: dict[tuple[Any, ...], dict[str, Any]] = {}

    def evaluate(ds: Any, cfg: PairedLipidConfig, **kwargs: Any) -> dict[str, Any]:
        # Deterministic evaluations repeat across axes (e.g. primary preset, primary seed,
        # anchors); compute each distinct (dataset, config, options) only once. The key holds
        # every option, and outer_seed=None means the primary seed, so equal runs share one entry.
        seed = kwargs.get("outer_seed")
        options = {**kwargs, "outer_seed": cfg.primary_seed if seed is None else int(seed)}
        key = (id(ds), cfg, tuple(sorted(options.items())))
        if key not in eval_memo:
            eval_memo[key] = evaluate_nested_cohort(ds, cfg=cfg, **kwargs)
        return eval_memo[key]

    base_ds, base_meta = resolve_experiment_dataset(
        use_synthetic=use_synthetic,
        approve_cohort_evaluation=approve_cohort_evaluation,
        clinical_master_path=clinical_master_path,
        subset_mode="all_eligible",
        compact_grid=compact_grid,
        cfg=base_cfg,
    )

    # 1. Covariate preset and subset mode sensitivity
    sens_rows: list[dict[str, Any]] = []
    for preset in COVARIATE_PRESETS:
        cfg_p = replace(base_cfg, covariate_preset=preset)
        res_p = evaluate(base_ds, cfg_p)
        sens_rows.append(
            {
                "sensitivity_axis": "covariate_preset",
                "setting": preset,
                "n_participants": base_ds.n_participants,
                **_paired_metrics(res_p),
            }
        )

    for s_mode in SUBSET_MODES:
        ds_sub, _ = resolve_experiment_dataset(
            use_synthetic=use_synthetic,
            approve_cohort_evaluation=approve_cohort_evaluation,
            clinical_master_path=clinical_master_path,
            subset_mode=s_mode,
            compact_grid=compact_grid,
            cfg=base_cfg,
        )
        res_sub = evaluate(
            ds_sub,
            base_cfg,
            single_sample_downstream=(s_mode == "single_sample"),
        )
        sens_rows.append(
            {
                "sensitivity_axis": "subset_mode",
                "setting": s_mode,
                "n_participants": ds_sub.n_participants,
                **_paired_metrics(res_sub),
            }
        )

    # 2. Additional fixed outer partition seeds sensitivity (dependent folds, never selected for superiority)
    seed_rows: list[dict[str, Any]] = []
    for seed in (base_cfg.primary_seed, *base_cfg.sensitivity_seeds):
        res_s = evaluate(base_ds, base_cfg, outer_seed=seed)
        seed_rows.append(
            {
                "outer_seed": int(seed),
                "role": "primary_locked" if seed == base_cfg.primary_seed else "sensitivity_only",
                **_paired_metrics(res_s),
            }
        )

    # 3. Secondary endpoints (HDL-C, TG, platelet count) with complete inner selection per target
    sec_rows: list[dict[str, Any]] = []
    # Same eligible population and outer folds as the primary endpoint (design); a participant
    # missing this endpoint is left out of every fold's training and test error.
    for ep in base_cfg.secondary_endpoints:
        res_ep = evaluate(base_ds, base_cfg, endpoint=ep, allow_missing_outcome=True)
        sec_rows.append(
            {
                "endpoint": ep,
                "family_role": "secondary_descriptive",
                "n_participants": base_ds.n_participants - res_ep["n_missing_outcome"],
                "n_missing_endpoint": res_ep["n_missing_outcome"],
                **_paired_metrics(res_ep, with_mae=True),
            }
        )

    # 4. Continuous exploratory anchor associations
    primary_eval = evaluate(base_ds, base_cfg)
    anchor_df = summarize_anchor_associations(
        base_ds,
        primary_eval,
        n_resamples=int(anchor_cfg["n_resamples"]),
        seed=int(anchor_cfg["seed"]),
        cfg=base_cfg,
    )

    sens_path = target_dir / "covariate_and_subset_sensitivity.csv"
    seeds_path = target_dir / "outer_partition_seed_sensitivity.csv"
    sec_path = target_dir / "secondary_endpoints_summary.csv"
    anchor_path = target_dir / "anchor_associations_continuous.csv"

    write_csv_atomic(pd.DataFrame(sens_rows), sens_path, index=False)
    write_csv_atomic(pd.DataFrame(seed_rows), seeds_path, index=False)
    write_csv_atomic(pd.DataFrame(sec_rows), sec_path, index=False)
    write_csv_atomic(anchor_df, anchor_path, index=False)
    prov_path = write_run_provenance(
        target_dir,
        "sensitivity_and_secondary",
        config_path=config_path,
        cfg=base_cfg,
        settings={"covariate_presets": list(COVARIATE_PRESETS), "subset_modes": list(SUBSET_MODES), "compact_grid": compact_grid},
        source_meta=base_meta,
    )

    return {
        "run_provenance_sensitivity_and_secondary": prov_path,
        "covariate_and_subset_sensitivity": sens_path,
        "outer_partition_seed_sensitivity": seeds_path,
        "secondary_endpoints_summary": sec_path,
        "anchor_associations_continuous": anchor_path,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run covariate/subset/seed sensitivity analyses and secondary endpoints."
    )
    add_cohort_stage_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_sensitivity_and_secondary(
        use_synthetic=not args.cohort,
        approve_cohort_evaluation=args.approve_cohort_evaluation,
        clinical_master_path=args.clinical_master,
        output_dir=args.output_dir,
        compact_grid=args.compact_grid,
        config_path=args.config,
    )
    for key, path in outputs.items():
        print(f"[run_sensitivity_and_secondary] wrote {key}: {path}")


if __name__ == "__main__":
    main()

