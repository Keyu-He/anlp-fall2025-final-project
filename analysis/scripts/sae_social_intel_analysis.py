import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def load_env_level_features(
    sae_path: Path,
) -> Tuple[np.ndarray, List[str], List[str], np.ndarray]:
    """Aggregate SAE features and rewards at environment level.

    SAE is conceptually very high dimensional (~131k features), but the log only
    stores the top-N active features per request (sae_top_indices / values).
    We aggregate those sparsely and then build a dense matrix over the union of
    all active feature indices.

    Returns:
        X: (num_envs, num_active_features) SAE feature matrix
           (mean activation per feature per env)
        env_ids: list of environment ids (len = num_envs)
        dims: list of reward dimensions (e.g., overall_score, believability, ...)
        Y: (num_envs, num_dims) reward matrix for agent2
    """
    # env_id -> feature_index -> (sum, count)
    env2_feat_stats: Dict[str, Dict[int, Tuple[float, int]]] = defaultdict(
        lambda: defaultdict(lambda: (0.0, 0))
    )
    env2_dim_sum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    env2_count: Dict[str, int] = defaultdict(int)

    with sae_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task_info = rec.get("task_info") or {}
            env_id = task_info.get("environment_id")
            if not env_id:
                continue

            rewards = task_info.get("rewards") or []
            if len(rewards) < 2:
                continue

            agent2_reward = rewards[1]
            if isinstance(agent2_reward, (list, tuple)) and len(agent2_reward) == 2:
                overall, dims_dict = agent2_reward
                if not isinstance(dims_dict, dict):
                    continue
            else:
                continue

            dims_dict = {k: float(v) for k, v in dims_dict.items()}
            dims_dict["overall_score"] = float(overall)

            # SAE sparse tops
            top_idx = (
                rec.get("sae_top_feature_indices")
                or rec.get("sae_top_indices")
                or []
            )
            top_val = (
                rec.get("sae_top_feature_values") or rec.get("sae_top_values") or []
            )
            if not isinstance(top_idx, list) or not isinstance(top_val, list):
                continue

            for fi, fv in zip(top_idx, top_val):
                try:
                    fi_int = int(fi)
                    fv_float = float(fv)
                except (TypeError, ValueError):
                    continue
                s, c = env2_feat_stats[env_id][fi_int]
                env2_feat_stats[env_id][fi_int] = (s + fv_float, c + 1)

            env2_count[env_id] += 1
            for k, v in dims_dict.items():
                env2_dim_sum[env_id][k] += v

    if not env2_feat_stats:
        raise SystemExit(f"No env-level data found in {sae_path}")

    # Collect all dimension names (union over envs)
    dim_names_set = set()
    for dim_sums in env2_dim_sum.values():
        dim_names_set.update(dim_sums.keys())
    dims = sorted(dim_names_set)

    # Create a stable ordering of all active feature indices
    active_features: List[int] = sorted(
        {fi for feat_stats in env2_feat_stats.values() for fi in feat_stats.keys()}
    )
    feat_index_to_col = {fi: idx for idx, fi in enumerate(active_features)}

    env_ids: List[str] = []
    X_rows: List[np.ndarray] = []
    Y_rows: List[np.ndarray] = []

    for env_id, feat_stats in env2_feat_stats.items():
        if env_id not in env2_dim_sum:
            continue
        count = env2_count[env_id]
        if count <= 0:
            continue

        x = np.zeros(len(active_features), dtype=np.float32)
        for fi, (s, c) in feat_stats.items():
            col = feat_index_to_col.get(fi)
            if col is None or c <= 0:
                continue
            x[col] = s / c

        dim_sums = env2_dim_sum[env_id]
        y = np.array([dim_sums.get(d, 0.0) / count for d in dims], dtype=np.float32)

        env_ids.append(env_id)
        X_rows.append(x)
        Y_rows.append(y)

    X = np.vstack(X_rows)
    Y = np.vstack(Y_rows)
    return X, env_ids, dims, Y


def compute_correlations(
    X: np.ndarray, Y: np.ndarray, dims: List[str]
) -> List[Dict[str, Any]]:
    """Compute Pearson correlation between each SAE feature and each reward dim."""
    num_envs, num_feats = X.shape
    _, num_dims = Y.shape

    results: List[Dict[str, Any]] = []

    # Center once for efficiency
    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    # Standard deviations
    X_std = X_centered.std(axis=0, ddof=1)
    Y_std = Y_centered.std(axis=0, ddof=1)

    for dim_idx, dim in enumerate(dims):
        y_c = Y_centered[:, dim_idx]
        sy = Y_std[dim_idx]
        if sy == 0:
            # Constant target; correlations are undefined -> treat as 0
            for feat_idx in range(num_feats):
                results.append(
                    {
                        "feature_index": int(feat_idx),
                        "dimension": dim,
                        "pearson_corr": 0.0,
                    }
                )
            continue

        # covariance between each feature and this dim
        cov = (X_centered * y_c[:, None]).sum(axis=0) / (num_envs - 1)
        corr = np.zeros_like(cov)
        nonzero = X_std > 0
        corr[nonzero] = cov[nonzero] / (X_std[nonzero] * sy)

        for feat_idx in range(num_feats):
            results.append(
                {
                    "feature_index": int(feat_idx),
                    "dimension": dim,
                    "pearson_corr": float(corr[feat_idx]),
                }
            )

    return results


def run_kmeans(
    X: np.ndarray, Y: np.ndarray, dims: List[str], k: int
) -> List[Dict[str, Any]]:
    """Cluster env-level SAE features and summarize rewards per cluster."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("sklearn not installed; skipping clustering analysis.")
        return []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=0, n_init="auto")
    labels = km.fit_predict(X_scaled)

    clusters: List[Dict[str, Any]] = []
    for cid in range(k):
        mask = labels == cid
        n = int(mask.sum())
        if n == 0:
            continue
        y_mean = Y[mask].mean(axis=0)
        rec: Dict[str, Any] = {"cluster_id": cid, "num_envs": n}
        for dim, val in zip(dims, y_mean):
            rec[f"mean_{dim}"] = float(val)
        clusters.append(rec)

    return clusters


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze correlations between SAE features and social-intelligence "
            "reward dimensions; also perform simple clustering."
        )
    )
    parser.add_argument(
        "--sae-path",
        type=Path,
        default=None,
        help=(
            "Path to joined SAE log "
            "(default: results/sae/sae_server_logs_l15_k64_20251201_133447_all_only_joined_all.jsonl)"
        ),
    )
    parser.add_argument(
        "--kmeans-k",
        type=int,
        default=10,
        help="Number of clusters for KMeans (default: 10).",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Prefix (directory) for output CSV files "
            "(default: analysis/data/sae_social_intel)"
        ),
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    project_root = base_dir.parent

    if args.sae_path is None:
        sae_path = project_root / "results" / "sae" / (
            "sae_server_logs_l15_k64_20251201_133447_all_only_joined_all.jsonl"
        )
    else:
        sae_path = args.sae_path

    if args.output_prefix is None:
        out_dir = base_dir / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = out_dir / "sae_social_intel"
    else:
        prefix = args.output_prefix
        prefix.parent.mkdir(parents=True, exist_ok=True)

    if not sae_path.exists():
        raise SystemExit(f"SAE joined file not found: {sae_path}")

    print(f"Loading SAE+reward data from: {sae_path}")
    X, env_ids, dims, Y = load_env_level_features(sae_path)
    print(f"Loaded {len(env_ids)} environments, {X.shape[1]} SAE features.")

    # Correlations
    corr_records = compute_correlations(X, Y, dims)
    corr_path = prefix.with_name(prefix.name + "_correlations.csv")
    with corr_path.open("w", encoding="utf-8") as f:
        f.write("feature_index,dimension,pearson_corr\n")
        for rec in corr_records:
            f.write(
                f"{rec['feature_index']},{rec['dimension']},{rec['pearson_corr']}\n"
            )
    print(f"Wrote correlations to {corr_path}")

    # Clustering
    cluster_records = run_kmeans(X, Y, dims, args.kmeans_k)
    if cluster_records:
        cluster_path = prefix.with_name(prefix.name + "_clusters.csv")
        with cluster_path.open("w", encoding="utf-8") as f:
            # header
            header = ["cluster_id", "num_envs"] + [f"mean_{d}" for d in dims]
            f.write(",".join(header) + "\n")
            for rec in cluster_records:
                row = [str(rec.get(h, "")) for h in header]
                f.write(",".join(row) + "\n")
        print(f"Wrote cluster summaries to {cluster_path}")


if __name__ == "__main__":
    main()
