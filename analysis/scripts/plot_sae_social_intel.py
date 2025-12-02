import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def load_correlations(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_clusters(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def ensure_matplotlib():
    try:
        import matplotlib.pyplot  # type: ignore[unused-ignore]
    except ImportError as e:  # pragma: no cover - visual dependency
        raise SystemExit(
            "matplotlib is required for plotting. "
            "Install with `pip install matplotlib` and rerun."
        ) from e


def plot_top_correlations(
    corr_rows: List[Dict[str, str]],
    dims: List[str],
    k: int,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import]

    out_dir.mkdir(parents=True, exist_ok=True)

    for dim in dims:
        entries: List[Tuple[float, float, int]] = []
        for row in corr_rows:
            if row["dimension"] != dim:
                continue
            c = float(row["pearson_corr"])
            fi = int(row["feature_index"])
            entries.append((abs(c), c, fi))

        if not entries:
            continue

        entries.sort(reverse=True)
        top = entries[:k]

        abs_vals, corr_vals, feat_ids = zip(*top)
        x = range(len(feat_ids))

        plt.figure(figsize=(10, 4))
        bars = plt.bar(x, corr_vals)
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xticks(
            x,
            [str(fi) for fi in feat_ids],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        plt.ylabel("Pearson corr.")
        plt.xlabel("SAE feature index")
        plt.title(f"Top {k} |corr| SAE features for {dim}")

        # Color positive / negative differently
        for b, c in zip(bars, corr_vals):
            b.set_color("tab:red" if c > 0 else "tab:blue")

        plt.tight_layout()
        out_path = out_dir / f"sae_corr_top{str(k)}_{dim}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_corr_histograms(
    corr_rows: List[Dict[str, str]],
    dims: List[str],
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import]

    out_dir.mkdir(parents=True, exist_ok=True)

    for dim in dims:
        vals: List[float] = []
        for row in corr_rows:
            if row["dimension"] != dim:
                continue
            vals.append(float(row["pearson_corr"]))
        if not vals:
            continue

        plt.figure(figsize=(5, 4))
        plt.hist(vals, bins=40, color="tab:gray", alpha=0.8)
        plt.axvline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Pearson corr.")
        plt.ylabel("Count of SAE features")
        plt.title(f"Correlation distribution for {dim} (n={len(vals)})")
        plt.tight_layout()
        out_path = out_dir / f"sae_corr_hist_{dim}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_cluster_bars(
    cluster_rows: List[Dict[str, str]],
    dims: List[str],
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import]

    if not cluster_rows:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort clusters by mean overall_score (if present)
    try:
        cluster_rows = sorted(
            cluster_rows,
            key=lambda r: float(r.get("mean_overall_score", "0")),
            reverse=True,
        )
    except ValueError:
        pass

    cluster_ids = [int(r["cluster_id"]) for r in cluster_rows]

    for dim in dims:
        key = f"mean_{dim}"
        if key not in cluster_rows[0]:
            continue
        vals = [float(r.get(key, "0")) for r in cluster_rows]

        plt.figure(figsize=(8, 4))
        x = range(len(cluster_ids))
        plt.bar(x, vals, color="tab:green", alpha=0.8)
        plt.xticks(x, [str(cid) for cid in cluster_ids], rotation=0)
        plt.xlabel("Cluster id")
        plt.ylabel(f"Mean {dim}")
        plt.title(f"{dim} per SAE cluster")
        plt.tight_layout()
        out_path = out_dir / f"sae_clusters_{dim}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot basic interpretability figures from SAE correlation and cluster CSVs."
    )
    parser.add_argument(
        "--corr-path",
        type=Path,
        default=None,
        help="Path to correlations CSV (default: analysis/sae/sae_social_intel_correlations.csv)",
    )
    parser.add_argument(
        "--cluster-path",
        type=Path,
        default=None,
        help="Path to clusters CSV (default: analysis/sae/sae_social_intel_clusters.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save figures (default: analysis/figures/sae)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k features by |corr| to plot per dimension (default: 20).",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    project_root = base_dir.parent

    if args.corr_path is None:
        corr_path = base_dir / "sae" / "sae_social_intel_correlations.csv"
    else:
        corr_path = args.corr_path

    if args.cluster_path is None:
        cluster_path = base_dir / "sae" / "sae_social_intel_clusters.csv"
    else:
        cluster_path = args.cluster_path

    if args.out_dir is None:
        out_dir = base_dir / "figures" / "sae"
    else:
        out_dir = args.out_dir

    if not corr_path.exists():
        raise SystemExit(f"Correlation CSV not found: {corr_path}")

    ensure_matplotlib()

    corr_rows = load_correlations(corr_path)

    # 重点关注的维度
    focus_dims = [
        "overall_score",
        "believability",
        "relationship",
        "knowledge",
        "secret",
        "social_rules",
        "financial_and_material_benefits",
        "goal",
    ]

    print(f"Loaded {len(corr_rows)} correlation rows from {corr_path}")
    print(f"Saving figures to {out_dir}")

    plot_top_correlations(corr_rows, focus_dims, args.top_k, out_dir)
    plot_corr_histograms(corr_rows, focus_dims, out_dir)

    if cluster_path.exists():
        cluster_rows = load_clusters(cluster_path)
        plot_cluster_bars(cluster_rows, ["overall_score", "social_rules", "relationship"], out_dir)
    else:
        print(f"No cluster CSV found at {cluster_path}; skipping cluster plots.")


if __name__ == "__main__":
    main()

