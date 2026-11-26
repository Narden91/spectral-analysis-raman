"""Step 4: Sensitivity analyses, secondary laboratory endpoints, and anchor associations.

Produces:
- ``covariate_and_subset_sensitivity.csv``
- ``outer_partition_seed_sensitivity.csv``
- ``secondary_endpoints_summary.csv``
- ``anchor_associations_continuous.csv``
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript_ml import (  # noqa: E402
    add_cohort_stage_args,
    ensure_output_dir,
    load_manuscript_config,
    print_stage_outputs,
    resolve_experiment_dataset,
    write_run_provenance,
)
from src.analysis.io_utils import write_csv_atomic  # noqa: E402
from src.analysis.paired_lipid.cohort import (  # noqa: E402
    COVARIATE_PRESETS,
    SUBSET_MODES,
    CohortDataset,
    PairedLipidConfig,
    SubsetMode,
)
from src.analysis.paired_lipid.nested_cv import (  # noqa: E402
    INTERVAL_METHODS,
    IntervalMethod,
    SpectralTrajectory,
    build_spectral_trajectory,
    evaluate_nested_cohort,
    run_multiplicity_bootstrap,
    summarize_anchor_associations,
)


def _summarize_strongest_comparator_name(res: Mapping[str, Any]) -> str:
    fs = res.get("fold_settings")
    if not isinstance(fs, pd.DataFrame) or fs.empty or "strongest_comparator_name" not in fs.columns:
        return ""
    names = [str(x) for x in fs["strongest_comparator_name"].tolist() if str(x)]
    if not names:
        return ""
    counts = Counter(names)
    if len(counts) == 1:
        return next(iter(counts))
    return ";".join(f"{name}({cnt})" for name, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _paired_metrics(
    res: dict[str, Any],
    *,
    with_mae: bool = False,
    primary_delta_mse: float | None = None,
) -> dict[str, Any]:
    """Paired-model and comparator summary columns shared by every sensitivity table."""
    comps = res["comparisons_vs_covariate_ridge"]
    m_pair = comps["paired_linear_ridge"]
    delta_pair = float(m_pair["delta_mse"])
    attenuation = (
        float(primary_delta_mse - delta_pair)
        if (primary_delta_mse is not None and math.isfinite(primary_delta_mse) and math.isfinite(delta_pair))
        else float("nan")
    )
    row: dict[str, Any] = {
        "delta_mse_paired": delta_pair,
        "relative_mse_reduction": float(m_pair["relative_mse_reduction"]),
        "rmse_paired": float(m_pair["rmse_model"]),
    }
    if with_mae:
        row["mae_paired"] = float(m_pair["mae_model"])
    row["delta_vs_strongest_comparator"] = float(res["paired_vs_strongest_comparator"]["delta_mse"])
    row["mse_covariate"] = float(m_pair.get("mse_covariate", float("nan")))
    row["mse_paired"] = float(m_pair.get("mse_model", float("nan")))
    row["mse_pca"] = float(comps.get("pca_ridge", {}).get("mse_model", float("nan")))
    row["mse_pls"] = float(comps.get("pls_regression", {}).get("mse_model", float("nan")))
    row["mse_fixed_band"] = float(comps.get("fixed_band_ridge", {}).get("mse_model", float("nan")))
    row["strongest_comparator_name"] = _summarize_strongest_comparator_name(res)
    row["attenuation_vs_primary_delta_mse"] = attenuation
    return row


def _dataset_memo_key(subset_mode: SubsetMode, ds: CohortDataset, source_meta: Mapping[str, Any]) -> tuple[Any, ...]:
    source_meta_key = (
        str(source_meta.get("data_mode", "")),
        str(source_meta.get("source_identity", "")),
        str(source_meta.get("store_manifest_hash", "")),
        source_meta.get("synthetic_seed"),
        source_meta.get("compact_grid"),
        ds.n_participants,
    )
    return (subset_mode, source_meta_key)


def run_sensitivity_and_secondary(
    *,
    use_synthetic: bool = False,
    approve_cohort_evaluation: bool = False,
    clinical_master_path: str | Path = PROJECT_ROOT / "data" / "clinical_master.csv",
    output_dir: str | Path | None = None,
    compact_grid: bool = False,
    config_path: str | Path | None = None,
    n_bootstraps: int = 0,
    interval_method: IntervalMethod | None = None,
    use_gpu: bool | None = None,
    batch_size: int | None = None,
    n_jobs: int | None = None,
    primary_state: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Run the configured sensitivity analyses, secondary endpoints, and anchor summaries."""
    target_dir = ensure_output_dir(output_dir)
    base_cfg, raw_cfg = load_manuscript_config(config_path)
    anchor_cfg = raw_cfg["anchors"]
    bootstrap_seed = int(raw_cfg["seeds"]["bootstrap_seed"])
    eff_interval_method: IntervalMethod = (
        raw_cfg["calibration"].get("interval_method", "studentized_t")
        if interval_method is None
        else interval_method
    )
    if eff_interval_method not in INTERVAL_METHODS:
        raise ValueError(f"Unsupported interval_method={eff_interval_method!r}; expected one of {INTERVAL_METHODS}.")
    if n_bootstraps < 0:
        raise ValueError(f"n_bootstraps must be >= 0; got {n_bootstraps}.")

    eval_memo: dict[tuple[Any, ...], dict[str, Any]] = {}
    boot_memo: dict[tuple[Any, ...], dict[str, Any]] = {}
    resolved_datasets: dict[SubsetMode, tuple[CohortDataset, dict[str, Any]]] = {}

    def get_dataset(mode: SubsetMode) -> tuple[CohortDataset, dict[str, Any]]:
        if mode not in resolved_datasets:
            resolved_datasets[mode] = resolve_experiment_dataset(
                use_synthetic=use_synthetic,
                approve_cohort_evaluation=approve_cohort_evaluation,
                clinical_master_path=clinical_master_path,
                subset_mode=mode,
                compact_grid=compact_grid,
                cfg=base_cfg,
                readiness=raw_cfg["readiness"],
                config_path=config_path,
            )
        return resolved_datasets[mode]

    def evaluate(
        subset_mode: SubsetMode,
        ds: CohortDataset,
        source_meta: Mapping[str, Any],
        cfg: PairedLipidConfig,
        *,
        endpoint: str | None = None,
        outer_seed: int | None = None,
        single_sample_downstream: bool = False,
        allow_missing_outcome: bool = False,
        compute_diagnostics: bool = False,
        spectral_trajectory: SpectralTrajectory | None = None,
    ) -> dict[str, Any]:
        # Deterministic evaluations repeat across axes (primary preset, all_eligible subset,
        # primary seed, and anchor associations); normalize every option into a typed key so
        # identical runs share one evaluation without relying on object memory addresses.
        resolved_endpoint = cfg.primary_endpoint if endpoint is None else str(endpoint)
        resolved_seed = cfg.primary_seed if outer_seed is None else int(outer_seed)
        ds_key = _dataset_memo_key(subset_mode, ds, source_meta)
        key = (
            ds_key,
            cfg,
            resolved_endpoint,
            resolved_seed,
            bool(single_sample_downstream),
            bool(allow_missing_outcome),
            bool(compute_diagnostics),
        )
        if key not in eval_memo:
            eval_kwargs: dict[str, Any] = {
                "cfg": cfg,
                "endpoint": resolved_endpoint,
                "outer_seed": resolved_seed,
                "single_sample_downstream": single_sample_downstream,
                "allow_missing_outcome": allow_missing_outcome,
                "compute_diagnostics": compute_diagnostics,
            }
            if spectral_trajectory is not None:
                eval_kwargs["spectral_trajectory"] = spectral_trajectory
            eval_memo[key] = evaluate_nested_cohort(ds, **eval_kwargs)
        return eval_memo[key]

    def bootstrap_ci_fields(
        subset_mode: SubsetMode,
        ds: CohortDataset,
        source_meta: Mapping[str, Any],
        cfg: PairedLipidConfig,
        *,
        single_sample_downstream: bool = False,
        spectral_trajectory: SpectralTrajectory | None = None,
    ) -> dict[str, Any]:
        if n_bootstraps <= 0:
            return {}
        b_key = (_dataset_memo_key(subset_mode, ds, source_meta), cfg, bool(single_sample_downstream))
        if b_key not in boot_memo:
            boot_kwargs: dict[str, Any] = {
                "cfg": cfg,
                "n_bootstraps": n_bootstraps,
                "bootstrap_seed": bootstrap_seed,
                "interval_method": eff_interval_method,
                "single_sample_downstream": single_sample_downstream,
                "use_gpu": use_gpu,
                "batch_size": batch_size,
                "n_jobs": n_jobs,
            }
            if spectral_trajectory is not None:
                boot_kwargs["spectral_trajectory"] = spectral_trajectory
            boot_memo[b_key] = run_multiplicity_bootstrap(ds, **boot_kwargs)
        b_res = boot_memo[b_key]
        return {
            "interval_method": b_res["interval_method"],
            "interval_procedure_succeeded": b_res["interval_procedure_succeeded"],
            "ci_95_delta_mse_low": float(b_res["ci_95_delta_mse"][0]),
            "ci_95_delta_mse_high": float(b_res["ci_95_delta_mse"][1]),
            "ci_95_vs_comparator_low": float(b_res["ci_95_vs_comparator"][0]),
            "ci_95_vs_comparator_high": float(b_res["ci_95_vs_comparator"][1]),
            "ci_95_strongest_vs_covariate_low": float(b_res["ci_95_strongest_vs_covariate"][0]),
            "ci_95_strongest_vs_covariate_high": float(b_res["ci_95_strongest_vs_covariate"][1]),
        }

    base_ds, base_meta = get_dataset("all_eligible")
    if primary_state is not None:
        if not isinstance(primary_state, Mapping):
            raise ValueError("primary_state must be a mapping when provided.")
        required_state_keys = ("cfg", "dataset", "source_meta", "eval_res")
        missing_state_keys = [k for k in required_state_keys if k not in primary_state]
        if missing_state_keys:
            raise ValueError(f"primary_state is missing required keys: {missing_state_keys}.")
        st_cfg = primary_state["cfg"]
        st_ds = primary_state["dataset"]
        st_meta = primary_state["source_meta"]
        st_eval = primary_state["eval_res"]
        if not isinstance(st_ds, CohortDataset):
            raise ValueError("primary_state['dataset'] must be a CohortDataset.")
        if not isinstance(st_meta, Mapping):
            raise ValueError("primary_state['source_meta'] must be a mapping.")
        if not isinstance(st_eval, Mapping):
            raise ValueError("primary_state['eval_res'] must be a mapping.")
        if st_cfg != base_cfg:
            raise ValueError("primary_state['cfg'] does not match sensitivity configuration.")
        if _dataset_memo_key("all_eligible", st_ds, st_meta) != _dataset_memo_key(
            "all_eligible", base_ds, base_meta
        ):
            raise ValueError("primary_state dataset/source_meta does not match sensitivity all_eligible dataset.")
        if str(st_ds.manifest_summary.get("subset_mode", "")) != "all_eligible":
            raise ValueError("primary_state['dataset'] must have subset_mode == 'all_eligible'.")
        if st_ds is not base_ds and not np.array_equal(st_ds.participant_ids, base_ds.participant_ids):
            raise ValueError("primary_state['dataset'] participant_ids do not match all_eligible dataset.")
        st_outer_seed = int(primary_state.get("outer_seed", base_cfg.primary_seed))
        if st_outer_seed != int(base_cfg.primary_seed):
            raise ValueError(
                f"primary_state['outer_seed'] ({st_outer_seed}) does not match primary_seed ({base_cfg.primary_seed})."
            )
        st_endpoint = str(primary_state.get("endpoint", base_cfg.primary_endpoint))
        if st_endpoint != str(base_cfg.primary_endpoint):
            raise ValueError(
                f"primary_state['endpoint'] ({st_endpoint!r}) does not match primary_endpoint ({base_cfg.primary_endpoint!r})."
            )
        if bool(primary_state.get("single_sample_downstream", False)) is not False:
            raise ValueError("primary_state['single_sample_downstream'] must be False for all_eligible reuse.")
        if bool(primary_state.get("allow_missing_outcome", False)) is not False:
            raise ValueError("primary_state['allow_missing_outcome'] must be False for all_eligible reuse.")
        required_eval_keys = (
            "comparisons_vs_covariate_ridge",
            "paired_vs_strongest_comparator",
            "oof_spectral_contributions",
        )
        missing_eval_keys = [k for k in required_eval_keys if k not in st_eval]
        if missing_eval_keys:
            raise ValueError(f"primary_state['eval_res'] is missing required keys: {missing_eval_keys}.")

        base_ds_key = _dataset_memo_key("all_eligible", base_ds, base_meta)
        eval_memo[
            (
                base_ds_key,
                base_cfg,
                base_cfg.primary_endpoint,
                int(base_cfg.primary_seed),
                False,
                False,
                False,
            )
        ] = dict(st_eval)

        if (
            n_bootstraps > 0
            and "boot_res" in primary_state
            and primary_state["boot_res"] is not None
            and primary_state.get("n_bootstraps") == n_bootstraps
            and int(primary_state.get("bootstrap_seed", -1)) == bootstrap_seed
            and primary_state.get("interval_method") == eff_interval_method
            and primary_state.get("use_gpu") == use_gpu
            and primary_state.get("batch_size") == batch_size
        ):
            boot_res_obj = primary_state["boot_res"]
            if not isinstance(boot_res_obj, Mapping):
                raise ValueError("primary_state['boot_res'] must be a mapping when provided.")
            required_boot_keys = (
                "interval_method",
                "interval_procedure_succeeded",
                "ci_95_delta_mse",
                "ci_95_vs_comparator",
                "ci_95_strongest_vs_covariate",
            )
            missing_boot_keys = [k for k in required_boot_keys if k not in boot_res_obj]
            if missing_boot_keys:
                raise ValueError(f"primary_state['boot_res'] is missing required keys: {missing_boot_keys}.")
            boot_memo[(base_ds_key, base_cfg, False)] = dict(boot_res_obj)

    base_traj: SpectralTrajectory | None = build_spectral_trajectory(
        base_ds,
        cfg=base_cfg,
        endpoint=base_cfg.primary_endpoint,
        outer_seed=base_cfg.primary_seed,
        single_sample_downstream=False,
        allow_missing_outcome=False,
        n_bootstraps=n_bootstraps,
        bootstrap_seed=bootstrap_seed,
        use_gpu=use_gpu,
        batch_size=batch_size,
        n_jobs=n_jobs,
        include_point=True,
    )
    primary_eval = evaluate(
        "all_eligible",
        base_ds,
        base_meta,
        base_cfg,
        spectral_trajectory=base_traj,
    )
    primary_delta_mse = float(primary_eval["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"])

    # 1. Covariate preset and subset mode sensitivity
    sens_rows: list[dict[str, Any]] = []
    for preset in COVARIATE_PRESETS:
        cfg_p = replace(base_cfg, covariate_preset=preset)
        res_p = evaluate(
            "all_eligible",
            base_ds,
            base_meta,
            cfg_p,
            spectral_trajectory=base_traj,
        )
        sens_rows.append(
            {
                "sensitivity_axis": "covariate_preset",
                "setting": preset,
                "n_participants": base_ds.n_participants,
                **_paired_metrics(res_p, primary_delta_mse=primary_delta_mse),
                **bootstrap_ci_fields(
                    "all_eligible",
                    base_ds,
                    base_meta,
                    cfg_p,
                    spectral_trajectory=base_traj,
                ),
            }
        )
    base_traj = None

    for s_mode in SUBSET_MODES:
        ds_sub, meta_sub = get_dataset(s_mode)
        is_single = s_mode == "single_sample"
        res_sub = evaluate(
            s_mode,
            ds_sub,
            meta_sub,
            base_cfg,
            single_sample_downstream=is_single,
        )
        sens_rows.append(
            {
                "sensitivity_axis": "subset_mode",
                "setting": s_mode,
                "n_participants": ds_sub.n_participants,
                **_paired_metrics(res_sub, primary_delta_mse=primary_delta_mse),
                **bootstrap_ci_fields(s_mode, ds_sub, meta_sub, base_cfg, single_sample_downstream=is_single),
            }
        )

    # 2. Additional fixed outer partition seeds sensitivity (dependent folds, never selected for superiority)
    seed_rows: list[dict[str, Any]] = []
    for seed in (base_cfg.primary_seed, *base_cfg.sensitivity_seeds):
        res_s = evaluate("all_eligible", base_ds, base_meta, base_cfg, outer_seed=seed)
        seed_rows.append(
            {
                "outer_seed": int(seed),
                "role": "primary_locked" if seed == base_cfg.primary_seed else "sensitivity_only",
                **_paired_metrics(res_s, primary_delta_mse=primary_delta_mse),
            }
        )

    # 3. Secondary endpoints (HDL-C, TG, platelet count) with complete inner selection per target
    sec_rows: list[dict[str, Any]] = []
    # Same eligible population and outer folds as the primary endpoint (design); a participant
    # missing this endpoint is left out of every fold's training and test error.
    for ep in base_cfg.secondary_endpoints:
        res_ep = evaluate(
            "all_eligible",
            base_ds,
            base_meta,
            base_cfg,
            endpoint=ep,
            allow_missing_outcome=True,
        )
        sec_rows.append(
            {
                "endpoint": ep,
                "family_role": "secondary_descriptive",
                "n_participants": base_ds.n_participants - res_ep["n_missing_outcome"],
                "n_missing_endpoint": res_ep["n_missing_outcome"],
                **_paired_metrics(res_ep, with_mae=True, primary_delta_mse=None),
            }
        )

    # 4. Continuous exploratory anchor associations
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
        settings={
            "covariate_presets": list(COVARIATE_PRESETS),
            "subset_modes": list(SUBSET_MODES),
            "compact_grid": compact_grid,
            "n_bootstraps": n_bootstraps,
            "interval_method": eff_interval_method,
            "use_gpu": use_gpu,
            "batch_size": batch_size,
        },
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
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=0,
        help="Optional participant-multiplicity bootstrap draws per covariate/subset sensitivity setting.",
    )
    parser.add_argument(
        "--interval-method",
        choices=INTERVAL_METHODS,
        default=None,
        help="Bootstrap interval estimator ('studentized_t' or 'percentile'); defaults to config.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batched bootstrap chunk size.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Optional worker count for CPU bootstrap draws.")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--use-gpu", dest="use_gpu", action="store_true", default=None, help="Force GPU execution.")
    gpu_group.add_argument("--no-gpu", dest="use_gpu", action="store_false", help="Force CPU execution.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_sensitivity_and_secondary(
        use_synthetic=args.synthetic,
        approve_cohort_evaluation=args.approve_cohort_evaluation,
        clinical_master_path=args.clinical_master,
        output_dir=args.output_dir,
        compact_grid=args.compact_grid,
        config_path=args.config,
        n_bootstraps=args.bootstraps,
        interval_method=args.interval_method,
        use_gpu=args.use_gpu,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
    )
    print_stage_outputs("run_sensitivity_and_secondary", outputs)


if __name__ == "__main__":
    main()
