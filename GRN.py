#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a Gephi-ready Gene Regulatory Network (GRN) from TransformerGP SHAP outputs.

Reads SHAP summary files produced by train.save_shap_values():
  {model_type}.{phenotype}_shap_values_exp.csv
  {model_type}.{phenotype}_shap_values_snp.csv

Builds:
- nodes.csv: Id, Label, Type(gene/snp), Importance, Size
- edges.csv: Source, Target, Weight

Edge weights are correlation-weighted:
- gene-gene: |corr(expr_i, expr_j)| * (imp_i + imp_j)/2
- snp-gene (optional): |corr(snp_k, expr_i)| * sqrt(imp_snp_k * imp_gene_i)

Then import into Gephi, run Modularity (or Leiden/Louvain), color by cluster,
size by Importance/Size, and edge thickness by Weight.
"""

import argparse
import os
import numpy as np
import pandas as pd

from config import EXP_FILE, PHENOTYPES_FILE, RESULTS_DIR


def _read_shap_summary(path: str) -> pd.DataFrame:
    """Read SHAP summary (index=Feature, columns include abs_mean). Returns df with column 'importance'."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"SHAP file not found: {path}")
    df = pd.read_csv(path, index_col=0)
    df.columns = [str(c).strip() for c in df.columns]

    candidates = [c for c in df.columns if c.lower() in ("abs_mean", "absmean", "mean_abs", "importance", "imp")]
    if not candidates:
        if df.shape[1] == 1:
            imp_col = df.columns[0]
        else:
            raise ValueError(
                f"Cannot find an importance column in {path}. "
                f"Expected abs_mean/mean_abs. Got columns: {df.columns.tolist()}"
            )
    else:
        imp_col = candidates[0]

    out = df[[imp_col]].copy()
    out.columns = ["importance"]
    out.index = out.index.astype(str)
    out["importance"] = pd.to_numeric(out["importance"], errors="coerce").fillna(0.0)
    return out.sort_values("importance", ascending=False)


def _load_expression_matrix_for_features(phenotype: str, features: list[str]) -> pd.DataFrame:
    """Load EXP_FILE, filter samples with phenotype, return samples x selected features."""
    ph = pd.read_csv(PHENOTYPES_FILE)
    ph = ph[ph[phenotype].notna()][["ID", phenotype]]

    exp = pd.read_csv(EXP_FILE)
    exp = exp[exp["ID"].isin(ph["ID"])].set_index("ID")
    exp = exp.select_dtypes(include=[np.number])

    existing = [f for f in features if f in exp.columns]
    missing = sorted(set(features) - set(existing))
    if not existing:
        raise ValueError("None of the requested EXP features are present in EXP_FILE.")
    if missing:
        print(f"[WARN] {len(missing)} EXP features not found in EXP_FILE. Skipping them.")

    X = exp[existing].replace([np.inf, -np.inf], np.nan)
    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
    return X


def _load_snp_matrix_for_features(phenotype: str, features: list[str]) -> pd.DataFrame:
    """
    Load SNP file by your codebase convention:
    /data1/wangchengrui/final_results/eqtl/rice4k_219/{phenotype}.csv
    """
    SNP_FILE = f"/data1/wangchengrui/final_results/eqtl/rice4k_219/{phenotype}.csv"

    ph = pd.read_csv(PHENOTYPES_FILE)
    ph = ph[ph[phenotype].notna()][["ID", phenotype]]

    snp = pd.read_csv(SNP_FILE)
    non_numeric_columns = ["FID", "PAT", "MAT", "SEX", "PHENOTYPE"]
    snp = snp.drop(columns=[c for c in non_numeric_columns if c in snp.columns], errors="ignore")
    snp = snp[snp["ID"].isin(ph["ID"])].set_index("ID")
    snp = snp.select_dtypes(include=[np.number])

    existing = [f for f in features if f in snp.columns]
    missing = sorted(set(features) - set(existing))
    if not existing:
        raise ValueError("None of the requested SNP features are present in the SNP file.")
    if missing:
        print(f"[WARN] {len(missing)} SNP features not found in SNP file. Skipping them.")

    X = snp[existing].replace([np.inf, -np.inf], np.nan)
    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
    return X


def _corr_edges_from_matrix(
    X: pd.DataFrame,
    importance: pd.Series,
    corr_threshold: float,
    max_edges: int | None,
    weight_mode: str = "mean",
) -> pd.DataFrame:
    """Undirected edges among columns of X based on Pearson correlation."""
    cols = X.columns.tolist()
    corr = X.corr(method="pearson").fillna(0.0).values

    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = abs(float(corr[i, j]))
            if c < corr_threshold:
                continue
            a, b = cols[i], cols[j]
            imp_i = float(importance.get(a, 0.0))
            imp_j = float(importance.get(b, 0.0))
            if weight_mode == "sqrt":
                imp_factor = float(np.sqrt(max(imp_i, 0.0) * max(imp_j, 0.0)))
            else:
                imp_factor = (imp_i + imp_j) / 2.0
            w = c * imp_factor
            if w <= 0:
                continue
            rows.append((a, b, w, c, imp_i, imp_j))

    edges = pd.DataFrame(rows, columns=["Source", "Target", "Weight", "AbsCorr", "ImpSource", "ImpTarget"])
    edges = edges.sort_values("Weight", ascending=False)
    if max_edges is not None and max_edges > 0 and edges.shape[0] > max_edges:
        edges = edges.iloc[:max_edges].copy()
    return edges


def _corr_bipartite_edges(
    X_left: pd.DataFrame,
    X_right: pd.DataFrame,
    imp_left: pd.Series,
    imp_right: pd.Series,
    corr_threshold: float,
    max_edges: int | None,
    weight_mode: str = "sqrt",
) -> pd.DataFrame:
    """Bipartite edges between columns of X_left and X_right based on correlation."""
    common = X_left.index.intersection(X_right.index)
    Xl = X_left.loc[common]
    Xr = X_right.loc[common]

    def zscore(df: pd.DataFrame) -> pd.DataFrame:
        mu = df.mean(axis=0)
        sd = df.std(axis=0).replace(0, 1.0)
        return (df - mu) / sd

    Xl_z = zscore(Xl)
    Xr_z = zscore(Xr)

    n = Xl_z.shape[0]
    corr_mat = (Xl_z.T.values @ Xr_z.values) / max(n - 1, 1)
    corr_mat = np.nan_to_num(corr_mat, nan=0.0, posinf=0.0, neginf=0.0)

    left_cols = Xl.columns.tolist()
    right_cols = Xr.columns.tolist()

    rows = []
    for i, a in enumerate(left_cols):
        for j, b in enumerate(right_cols):
            c = abs(float(corr_mat[i, j]))
            if c < corr_threshold:
                continue
            imp_a = float(imp_left.get(a, 0.0))
            imp_b = float(imp_right.get(b, 0.0))
            if weight_mode == "mean":
                imp_factor = (imp_a + imp_b) / 2.0
            else:
                imp_factor = float(np.sqrt(max(imp_a, 0.0) * max(imp_b, 0.0)))
            w = c * imp_factor
            if w <= 0:
                continue
            rows.append((a, b, w, c, imp_a, imp_b))

    edges = pd.DataFrame(rows, columns=["Source", "Target", "Weight", "AbsCorr", "ImpSource", "ImpTarget"])
    edges = edges.sort_values("Weight", ascending=False)
    if max_edges is not None and max_edges > 0 and edges.shape[0] > max_edges:
        edges = edges.iloc[:max_edges].copy()
    return edges


def build_grn(
    phenotype: str,
    model_type: str,
    top_exp: int,
    top_snp: int,
    include_snp_nodes: bool,
    corr_threshold_exp: float,
    corr_threshold_snp_gene: float,
    max_edges_exp: int | None,
    max_edges_snp_gene: int | None,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    exp_shap_path = os.path.join(RESULTS_DIR, f"{model_type}.{phenotype}_shap_values_exp.csv")
    snp_shap_path = os.path.join(RESULTS_DIR, f"{model_type}.{phenotype}_shap_values_snp.csv")

    exp_imp_df = _read_shap_summary(exp_shap_path)
    snp_imp_df = _read_shap_summary(snp_shap_path)

    exp_features = exp_imp_df.head(top_exp).index.tolist() if top_exp > 0 else exp_imp_df.index.tolist()
    snp_features = snp_imp_df.head(top_snp).index.tolist() if top_snp > 0 else snp_imp_df.index.tolist()

    exp_importance = exp_imp_df["importance"]
    snp_importance = snp_imp_df["importance"]

    exp_X = _load_expression_matrix_for_features(phenotype, exp_features)

    edges_gg = _corr_edges_from_matrix(
        exp_X, exp_importance,
        corr_threshold=corr_threshold_exp,
        max_edges=max_edges_exp,
        weight_mode="mean",
    )
    edges_gg["EdgeType"] = "gene-gene"

    if include_snp_nodes and (top_snp != 0):
        snp_X = _load_snp_matrix_for_features(phenotype, snp_features)
        edges_sg = _corr_bipartite_edges(
            X_left=snp_X, X_right=exp_X,
            imp_left=snp_importance, imp_right=exp_importance,
            corr_threshold=corr_threshold_snp_gene,
            max_edges=max_edges_snp_gene,
            weight_mode="sqrt",
        )
        edges_sg["EdgeType"] = "snp-gene"
        edges = pd.concat([edges_gg, edges_sg], axis=0, ignore_index=True)
    else:
        edges = edges_gg

    if edges.empty:
        raise ValueError(
            "No edges generated. Lower thresholds: --corr_thr_exp / --corr_thr_snp_gene, "
            "or increase --top_exp/--top_snp."
        )

    node_ids = pd.Index(pd.unique(edges[["Source", "Target"]].values.ravel("K"))).astype(str)

    node_type = []
    node_importance = []
    for nid in node_ids:
        if nid in exp_importance.index:
            node_type.append("gene")
            node_importance.append(float(exp_importance.get(nid, 0.0)))
        elif nid in snp_importance.index:
            node_type.append("snp")
            node_importance.append(float(snp_importance.get(nid, 0.0)))
        else:
            node_type.append("unknown")
            node_importance.append(0.0)

    nodes = pd.DataFrame({
        "Id": node_ids,
        "Label": node_ids,
        "Type": node_type,
        "Importance": node_importance,
    })

    imp = nodes["Importance"].astype(float).values
    size = np.log1p(np.maximum(imp, 0.0))
    if np.max(size) > 0:
        size = 5 + 45 * (size / np.max(size))
    else:
        size = np.ones_like(size) * 5
    nodes["Size"] = size

    nodes_path = os.path.join(out_dir, f"{model_type}.{phenotype}_nodes.csv")
    edges_path = os.path.join(out_dir, f"{model_type}.{phenotype}_edges.csv")
    edges_full_path = os.path.join(out_dir, f"{model_type}.{phenotype}_edges_full.csv")

    nodes.to_csv(nodes_path, index=False)
    edges[["Source", "Target", "Weight"]].to_csv(edges_path, index=False)
    edges.to_csv(edges_full_path, index=False)

    print(f"[OK] Nodes saved: {nodes_path}")
    print(f"[OK] Edges saved: {edges_path}")
    print(f"[OK] Edges(full) saved: {edges_full_path}")
    print("\nGephi:")
    print("  - Import nodes.csv (Nodes) + edges.csv (Edges)")
    print("  - Run Statistics -> Modularity (or Leiden/Louvain)")
    print("  - Color nodes by Modularity Class; size nodes by Size/Importance; edge thickness by Weight")


def main():
    ap = argparse.ArgumentParser(description="Build a GRN (Gephi-ready) from TransformerGP SHAP files.")
    ap.add_argument("--phenotype", default="Heading_date", help="Phenotype name (must match SHAP filenames).")
    ap.add_argument("--model_type", default="combined", choices=["exp", "snp", "combined"],
                    help="Prefix used in SHAP filenames: {model_type}.{phenotype}_shap_values_*.csv")
    ap.add_argument("--out_dir", default=None, help="Output dir (default: RESULTS_DIR/grn)")
    ap.add_argument("--top_exp", type=int, default=200, help="Top N EXP features by SHAP importance (0=all).")
    ap.add_argument("--top_snp", type=int, default=200, help="Top N SNP features by SHAP importance (0=all).")
    ap.add_argument("--include_snp", action="store_true", help="Include SNP nodes and SNP-gene edges.")
    ap.add_argument("--corr_thr_exp", type=float, default=0.3, help="Abs corr threshold for gene-gene edges.")
    ap.add_argument("--corr_thr_snp_gene", type=float, default=0.2, help="Abs corr threshold for SNP-gene edges.")
    ap.add_argument("--max_edges_exp", type=int, default=5000, help="Max gene-gene edges (0=no limit).")
    ap.add_argument("--max_edges_snp_gene", type=int, default=5000, help="Max SNP-gene edges (0=no limit).")

    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(RESULTS_DIR, "grn")
    max_edges_exp = None if args.max_edges_exp == 0 else args.max_edges_exp
    max_edges_snp_gene = None if args.max_edges_snp_gene == 0 else args.max_edges_snp_gene

    build_grn(
        phenotype=args.phenotype,
        model_type=args.model_type,
        top_exp=args.top_exp,
        top_snp=args.top_snp,
        include_snp_nodes=args.include_snp,
        corr_threshold_exp=args.corr_thr_exp,
        corr_threshold_snp_gene=args.corr_thr_snp_gene,
        max_edges_exp=max_edges_exp,
        max_edges_snp_gene=max_edges_snp_gene,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
