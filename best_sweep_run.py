import argparse
from typing import Any, Dict, List, Optional, Tuple

import wandb


TARGET_KEYS = [
    "latent_lr",
    "render_weight",
    "guided_N",
    "guided_eta",
    "flow_train_epochs",
]


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_sweep_path(entity: str, project: str, sweep_id_or_path: str) -> str:
    if "/" in sweep_id_or_path:
        return sweep_id_or_path
    return f"{entity}/{project}/{sweep_id_or_path}"


def _collect_scored_runs(runs: List[Any], metric_name: str) -> List[Tuple[float, Any]]:
    scored = []
    for run in runs:
        score = _to_float(run.summary.get(metric_name))
        if score is not None:
            scored.append((score, run))
    return scored


def _extract_params(run_config: Dict[str, Any]) -> Dict[str, Any]:
    params = {}
    for key in TARGET_KEYS:
        if key in run_config:
            params[key] = run_config[key]
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Report best sweep run by latent_psnr.")
    parser.add_argument("--entity", type=str, default="", help="W&B entity/user. Optional if sweep path is full.")
    parser.add_argument("--project", type=str, default="mnist-flow-matching", help="W&B project name.")
    parser.add_argument("--sweep", type=str, required=True, help="Sweep ID or full path entity/project/sweep_id.")
    parser.add_argument("--metric", type=str, default="latent_psnr", help="Summary metric name.")
    args = parser.parse_args()

    api = wandb.Api()
    sweep_path = _resolve_sweep_path(args.entity, args.project, args.sweep)
    sweep = api.sweep(sweep_path)

    scored_runs = _collect_scored_runs(sweep.runs, args.metric)
    if not scored_runs:
        print(f"No runs in sweep '{sweep_path}' have numeric '{args.metric}' values yet.")
        return

    best_score, best_run = max(scored_runs, key=lambda x: x[0])
    best_params = _extract_params(best_run.config)

    print(f"SWEEP={sweep_path}")
    print(f"BEST_RUN_ID={best_run.id}")
    print(f"BEST_RUN_NAME={best_run.name}")
    print(f"BEST_{args.metric.upper()}={best_score:.6f}")
    print("BEST_PARAMS=")
    for key in TARGET_KEYS:
        if key in best_params:
            print(f"  {key}: {best_params[key]}")


if __name__ == "__main__":
    main()
