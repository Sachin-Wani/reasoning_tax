"""tes_analysis.py
===============
Computes all metrics and generates all figures for:
  "The Reasoning Tax: Token Economics of LLM Reasoning
   Across Task Types and Deployment Contexts"

Usage:
    python tes_analysis.py --csv actuals_final.csv

Outputs (written to ./output/):
    tes_pairs.csv          — TES for every paired comparison
    benchmark_summary.csv  — Mean ± std TES per benchmark
    rcs_table.csv          — Reasoning Cost Share per reasoning run
    dcm_table.csv          — Deployment Cost Multiplier per on-prem model
    diminishing_returns.csv— Marginal TES across effort levels
    tes_a_gemini.csv       — TES-A values for unpaired Gemini entries
    quadrant_data.csv      — Data underlying Figure 5

    figure2_tes_distribution.png  — TES distribution across benchmarks
    figure3_diminishing_returns.png
    figure4_rcs_stacked.png
    figure5_quadrant.png
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe on all platforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import LogLocator, LogFormatter

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Benchmark ordering (difficulty / task-type axis for plots) ───────────────
BENCH_ORDER = [
    "AIME 2025",
    "IFBench",
    "LiveCodeBench",
    "HLE",
    "GPQA Diamond",
    "CritPt",
    "MMLU-Pro",
]

# Colour per benchmark (used consistently in all figures)
BENCH_COLOURS = {
    "AIME 2025":     "#2563EB",   # blue   — math inference chain
    "LiveCodeBench": "#16A34A",   # green  — code inference chain
    "IFBench":       "#9333EA",   # purple — instruction following
    "HLE":           "#D97706",   # amber  — expert multi-domain
    "GPQA Diamond":  "#DC2626",   # red    — science reasoning
    "CritPt":        "#0891B2",   # cyan   — extreme physics
    "MMLU-Pro":      "#6B7280",   # grey   — knowledge recall
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the evaluation CSV and add a derived column `gen_tokens`:

        gen_tokens = reasoning_tokens + output_tokens

    This is the denominator basis for TES.  Input tokens are excluded because
    they are determined by the benchmark prompt and evaluation harness, not by
    the model's reasoning mechanism.  Including them would make TES sensitive
    to prompt engineering choices rather than model reasoning behaviour.
    """
    df = pd.read_csv(csv_path)

    # Derived: generation tokens (what the model *produced*, excluding prompt)
    df["gen_tokens"] = df["reasoning_tokens"] + df["output_tokens"]

    # Normalise pair_id — treat NaN / empty string / 0 uniformly as unpaired
    df["pair_id"] = df["pair_id"].fillna("").astype(str).str.strip()
    df.loc[df["pair_id"] == "0", "pair_id"] = ""

    # Normalise on_prem_cost — treat NaN as 0
    df["on_prem_cost"] = df["on_prem_cost"].fillna(0.0)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — TES CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def calculate_tes_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TES for every complete paired comparison in the dataset.

    Formula
    -------
        TES(M_r, M_i, T) = [Acc(M_r, T) - Acc(M_i, T)] * 100
                           ─────────────────────────────────────
                           GenTok(M_r, T) / GenTok(M_i, T)

    where
        Acc(M, T)    = accuracy of model M on benchmark T (0–1 scale in CSV,
                       multiplied by 100 to express numerator in pp)
        GenTok(M, T) = reasoning_tokens + output_tokens  (see load_data)

    Numerator  — accuracy gain in *percentage points* (pp).
                 Using pp (not 0–1 decimal) gives interpretable TES magnitudes;
                 e.g. TES=6 means 6 pp of gain per unit of token-cost multiple.

    Denominator — generated-token cost multiplier: how many times more tokens
                  the reasoning variant consumed relative to the baseline.
                  A value of 3.5 means the reasoning model generated 3.5× more
                  tokens than the instruct baseline.

    Interpretation
    --------------
        TES > 1  : accuracy gain (pp) exceeds token-cost multiplier — strong
        0 < TES ≤ 1 : positive but marginal
        TES ≤ 0  : wasteful or harmful — reasoning costs tokens for no gain

    Variant labelling
    -----------------
        TES-Δ : both M_r and M_i share the same pair_id (same architecture,
                reasoning toggled on vs off)
        TES-A : M_i is the best instruct model for that benchmark (Gemini only
                in this dataset; computed separately in calculate_tes_a)

    Returns a DataFrame with one row per reasoning model per benchmark.
    """
    # Work only on rows that have a non-empty pair_id
    paired = df[(df["pair_id"] != "")].copy()

    records = []

    for bench in df["benchmark"].unique():
        bench_df = paired[paired["benchmark"] == bench]

        for pid, grp in bench_df.groupby("pair_id"):
            instruct_rows  = grp[grp["thinking"] == 0]
            reasoning_rows = grp[grp["thinking"] == 1]

            # Skip incomplete pairs
            if instruct_rows.empty or reasoning_rows.empty:
                continue

            # Use the single instruct row as baseline M_i
            mi = instruct_rows.iloc[0]

            # There may be multiple reasoning variants (high / xhigh / max)
            # — iterate over each one separately so we capture diminishing
            #   returns within a family
            for _, mr in reasoning_rows.iterrows():

                # ── Numerator: accuracy delta in percentage points ──────────
                delta_acc_pp = (mr["score"] - mi["score"]) * 100

                # ── Denominator: generated-token ratio ─────────────────────
                gen_tok_i = mi["gen_tokens"]
                gen_tok_r = mr["gen_tokens"]

                if gen_tok_i == 0:
                    # Cannot compute a ratio if baseline generated 0 tokens
                    continue

                gen_tok_mult = gen_tok_r / gen_tok_i

                # ── TES ────────────────────────────────────────────────────
                tes = delta_acc_pp / gen_tok_mult

                # ── Cost multiplier (for reference / DTES) ─────────────────
                cost_mult = (
                    mr["cost_total"] / mi["cost_total"]
                    if mi["cost_total"] > 0
                    else np.nan
                )

                records.append({
                    "benchmark":      bench,
                    "pair_id":        pid,
                    "model_r":        mr["model"],
                    "model_i":        mi["model"],
                    "family":         mr["family"],
                    "source":         mr["source"],
                    "acc_i_pct":      round(mi["score"] * 100, 2),
                    "acc_r_pct":      round(mr["score"] * 100, 2),
                    "delta_acc_pp":   round(delta_acc_pp, 3),
                    "gen_tok_i":      int(gen_tok_i),
                    "gen_tok_r":      int(gen_tok_r),
                    "gen_tok_mult":   round(gen_tok_mult, 4),
                    "TES":            round(tes, 4),
                    "cost_i":         round(mi["cost_total"], 6),
                    "cost_r":         round(mr["cost_total"], 6),
                    "cost_mult":      round(cost_mult, 4) if not np.isnan(cost_mult) else np.nan,
                    "reasoning_tok_r": int(mr["reasoning_tokens"]),
                    "output_tok_r":   int(mr["output_tokens"]),
                })

    result = pd.DataFrame(records)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — BENCHMARK SUMMARY (mean ± std TES)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_summary(
    tes_df: pd.DataFrame,
    ifbench_outlier_model: str = "Grok 4.20 0309",
) -> pd.DataFrame:
    """
    Compute mean ± standard deviation of TES per benchmark.

    The Grok 4.20 0309 entry on IFBench is excluded from IFBench statistics
    as documented in Section 4.5 of the paper: its instruct variant generated
    870,000 output tokens — an anomaly likely caused by the non-reasoning model
    producing unstructured verbose output — resulting in a near-unity cost
    multiplier and an inflated TES of ~30.  The pair is included in the full
    per-model table (Table A1) but excluded from aggregate statistics.

    Returns a summary DataFrame sorted by mean TES descending.
    """
    records = []

    for bench in BENCH_ORDER:
        sub = tes_df[tes_df["benchmark"] == bench].copy()

        # Total pairs before any exclusion
        n_total = len(sub)

        # Exclude outlier from IFBench stats
        if bench == "IFBench":
            sub_stats = sub[sub["model_r"] != ifbench_outlier_model]
            n_stats   = len(sub_stats)
            outlier_excluded = True
        else:
            sub_stats = sub
            n_stats   = n_total
            outlier_excluded = False

        if sub_stats.empty:
            continue

        mean_tes   = sub_stats["TES"].mean()
        std_tes    = sub_stats["TES"].std(ddof=1)   # sample std dev
        median_tes = sub_stats["TES"].median()
        min_tes    = sub_stats["TES"].min()
        max_tes    = sub_stats["TES"].max()

        records.append({
            "benchmark":         bench,
            "mean_TES":          round(mean_tes,  3),
            "std_TES":           round(std_tes,   3),
            "median_TES":        round(median_tes, 3),
            "min_TES":           round(min_tes,   3),
            "max_TES":           round(max_tes,   3),
            "n_pairs_stats":     n_stats,
            "n_pairs_total":     n_total,
            "outlier_excluded":  outlier_excluded,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — REASONING COST SHARE (RCS)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_rcs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Reasoning Cost Share for every reasoning-enabled model run.

    Formula
    -------
        RCS(M_r, T) = cost_reasoning / cost_total

    where
        cost_reasoning = (reasoning_tokens / 1,000,000) * price_per_million_output
        cost_total     = cost_input + cost_reasoning + cost_output

    RCS is expressed as a percentage (0–100).

    Interpretation
    --------------
        RCS close to 100% means virtually all inference spend goes into the
        thinking chain; the final visible answer costs almost nothing.
        RCS of 66% (the dataset minimum) means a third of the cost is the
        final answer — the most balanced observed in this study.

    Only reasoning-enabled rows (thinking == 1) with cost_total > 0 are
    included.  Instruct rows have cost_reasoning = 0 by definition and
    RCS = 0 trivially — including them adds no information.
    """
    reasoning = df[(df["thinking"] == 1) & (df["cost_total"] > 0)].copy()

    reasoning["RCS_pct"] = (
        reasoning["cost_reasoning"] / reasoning["cost_total"]
    ) * 100

    cols = [
        "model", "family", "benchmark", "thinking",
        "score", "reasoning_tokens", "output_tokens", "gen_tokens",
        "cost_reasoning", "cost_total", "RCS_pct",
        "price_per_million_output", "source",
    ]
    result = reasoning[cols].copy()
    result["RCS_pct"] = result["RCS_pct"].round(2)
    result = result.sort_values("RCS_pct", ascending=False).reset_index(drop=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — DEPLOYMENT COST MULTIPLIER (DCM)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_dcm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Deployment Cost Multiplier for every row that has an on-prem cost.

    Formula
    -------
        DCM(M, T) = cost_total_cloud / cost_total_onprem

    where
        cost_total_cloud  = cost_total column (cloud API pricing from AA)
        cost_total_onprem = on_prem_cost column (derived from B300 amortised
                            cost per second divided by measured throughput;
                            see Section 4.4)

    DCM > 1 always (cloud is always more expensive than on-prem in this
    dataset).  Higher DCM means a larger cost reduction from on-premises
    deployment.

    Both thinking=0 (instruct) and thinking=1 (reasoning) rows are included
    because DCM is a property of the model and hardware, not of whether
    reasoning is enabled.  This allows verification that DCM is consistent
    across reasoning modes for the same model.

    Rows with on_prem_cost == 0 are skipped (no on-prem data available).
    """
    onprem = df[df["on_prem_cost"] > 0].copy()

    # DCM: how many times cheaper is on-prem vs cloud for the same workload
    onprem["DCM"] = onprem["cost_total"] / onprem["on_prem_cost"]

    cols = [
        "model", "family", "benchmark", "thinking",
        "cost_total", "on_prem_cost", "DCM",
        "score", "source",
    ]
    result = onprem[cols].copy()
    result["DCM"] = result["DCM"].round(3)
    result = result.sort_values(["model", "benchmark", "thinking"]).reset_index(drop=True)
    return result


def dcm_model_summary(dcm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate DCM per model (mean across benchmarks and thinking modes).

    This is the summary used in Table 5 of the paper.  The consistency check
    (max_DCM - min_DCM) confirms that DCM is a stable property of the
    model-hardware pair rather than workload-dependent.
    """
    summary = (
        dcm_df.groupby("model")["DCM"]
        .agg(mean_DCM="mean", min_DCM="min", max_DCM="max", n="count")
        .round(2)
        .reset_index()
    )
    summary["DCM_range"] = (summary["max_DCM"] - summary["min_DCM"]).round(2)
    return summary.sort_values("mean_DCM", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — DIMINISHING RETURNS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def diminishing_returns(tes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify model families with multiple reasoning effort levels on the same
    benchmark and compute the marginal TES between consecutive effort levels.

    Marginal TES
    ------------
        For two reasoning variants r1 (lower effort) and r2 (higher effort)
        sharing the same pair_id and benchmark, the marginal TES of upgrading
        from r1 to r2 is:

            MarginalTES(r1→r2) = [Acc(r2) - Acc(r1)] * 100
                                  ─────────────────────────
                                  GenTok(r2) / GenTok(r1)

        A negative MarginalTES means accuracy *falls* when effort is increased
        — the reasoning saturation point has been exceeded.

    This function works on the already-computed tes_df (output of
    calculate_tes_pairs) and groups by pair_id + benchmark, finding cases
    where more than one reasoning variant exists.
    """
    # Find pair_id+benchmark combos with multiple reasoning rows
    multi = (
        tes_df.groupby(["benchmark", "pair_id"])
        .filter(lambda g: len(g) > 1)
    )

    records = []

    for (bench, pid), grp in multi.groupby(["benchmark", "pair_id"]):
        # Sort by generated tokens ascending (proxy for effort level)
        grp_sorted = grp.sort_values("gen_tok_r").reset_index(drop=True)

        for idx in range(len(grp_sorted) - 1):
            r1 = grp_sorted.iloc[idx]
            r2 = grp_sorted.iloc[idx + 1]

            # Marginal accuracy gain from r1 → r2
            marginal_delta_pp = r2["delta_acc_pp"] - r1["delta_acc_pp"]

            # Marginal token multiple: how many more tokens r2 uses vs r1
            if r1["gen_tok_r"] == 0:
                continue
            marginal_tok_mult = r2["gen_tok_r"] / r1["gen_tok_r"]

            marginal_tes = marginal_delta_pp / marginal_tok_mult

            records.append({
                "benchmark":          bench,
                "pair_id":            pid,
                "model_lower_effort": r1["model_r"],
                "model_higher_effort":r2["model_r"],
                "acc_lower_pct":      r1["acc_r_pct"],
                "acc_higher_pct":     r2["acc_r_pct"],
                "marginal_delta_pp":  round(marginal_delta_pp, 3),
                "TES_lower":          r1["TES"],
                "TES_higher":         r2["TES"],
                "gen_tok_lower":      r1["gen_tok_r"],
                "gen_tok_higher":     r2["gen_tok_r"],
                "marginal_tok_mult":  round(marginal_tok_mult, 3),
                "marginal_TES":       round(marginal_tes, 4),
                "accuracy_drops":     marginal_delta_pp < 0,
            })

    result = pd.DataFrame(records)
    result = result.sort_values("marginal_TES").reset_index(drop=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — TES-A FOR UNPAIRED MODELS (GEMINI)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_tes_a(df: pd.DataFrame, tes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TES-A for models that have no non-reasoning counterpart (pair_id
    is blank) — in this dataset, all Gemini entries fall into this category.

    TES-A uses the same formula as TES-Δ but M_i is the best-performing
    non-reasoning model on that benchmark, identified from the dataset.

        M_i = argmax Acc(M, T)  subject to  thinking(M) == 0

    This conflates the reasoning mechanism with general capability differences
    between model families (see Section 3.3 of the paper), but reflects the
    realistic deployment counterfactual: the practitioner's alternative is the
    best instruct model available, not a hypothetical reasoning-disabled version
    of the same frontier model.
    """
    # Identify unpaired reasoning rows (blank pair_id, thinking == 1)
    unpaired_r = df[(df["pair_id"] == "") & (df["thinking"] == 1)].copy()

    if unpaired_r.empty:
        return pd.DataFrame()

    records = []

    for bench in unpaired_r["benchmark"].unique():
        # Best instruct baseline for this benchmark
        instruct_pool = df[(df["benchmark"] == bench) & (df["thinking"] == 0)]
        if instruct_pool.empty:
            continue

        best_i_idx = instruct_pool["score"].idxmax()
        mi = instruct_pool.loc[best_i_idx]

        bench_unpaired = unpaired_r[unpaired_r["benchmark"] == bench]

        for _, mr in bench_unpaired.iterrows():
            delta_acc_pp = (mr["score"] - mi["score"]) * 100

            gen_tok_i = mi["gen_tokens"]
            gen_tok_r = mr["gen_tokens"]
            if gen_tok_i == 0:
                continue

            gen_tok_mult = gen_tok_r / gen_tok_i
            tes_a = delta_acc_pp / gen_tok_mult

            records.append({
                "benchmark":       bench,
                "model_r":         mr["model"],
                "family_r":        mr["family"],
                "model_i":         mi["model"],
                "family_i":        mi["family"],
                "acc_i_pct":       round(mi["score"] * 100, 2),
                "acc_r_pct":       round(mr["score"] * 100, 2),
                "delta_acc_pp":    round(delta_acc_pp, 3),
                "gen_tok_i":       int(gen_tok_i),
                "gen_tok_r":       int(gen_tok_r),
                "gen_tok_mult":    round(gen_tok_mult, 4),
                "TES_A":           round(tes_a, 4),
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — PER-QUESTION TOKEN AVERAGES (for Section 7.1 of paper)
# ─────────────────────────────────────────────────────────────────────────────

# Number of questions per benchmark (used to convert benchmark totals to
# per-question averages for the agentic context-window calculation in §7.1)
BENCHMARK_QUESTION_COUNTS = {
    "IFBench":       58,
    "MMLU-Pro":   12000,
    "GPQA Diamond": 198,
    "AIME 2025":     30,
    "LiveCodeBench": None,   # continuous harvest; no fixed count
    "HLE":         2500,
    "CritPt":        71,
}


def per_question_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average tokens per question for each model-benchmark entry by
    dividing aggregate token counts by the number of questions in the
    benchmark.

    This is used in Section 7.1 to derive the agentic context-window fill
    rate:  how many agentic steps before reasoning residue fills the context.

    Only benchmarks with a known fixed question count are included.
    """
    records = []
    for _, row in df.iterrows():
        n_q = BENCHMARK_QUESTION_COUNTS.get(row["benchmark"])
        if n_q is None or n_q == 0:
            continue
        records.append({
            "model":                    row["model"],
            "benchmark":                row["benchmark"],
            "thinking":                 row["thinking"],
            "n_questions":              n_q,
            "reasoning_tok_per_q":      row["reasoning_tokens"] / n_q,
            "output_tok_per_q":         row["output_tokens"]    / n_q,
            "gen_tok_per_q":            row["gen_tokens"]       / n_q,
            "context_steps_200k":       200_000 / (row["gen_tokens"] / n_q)
                                        if row["gen_tokens"] > 0 else np.nan,
        })
    result = pd.DataFrame(records)
    result = result.round(1)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — QUADRANT ANALYSIS (for Figure 5)
# ─────────────────────────────────────────────────────────────────────────────

def quadrant_analysis(
    tes_df: pd.DataFrame,
    tes_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Assign each reasoning model run to one of four deployment quadrants based
    on TES and cloud inference cost.

    TES threshold  : 1.0 (the boundary between strong and marginal)
    Cost threshold : median cloud cost across all reasoning runs in the dataset

    Quadrants
    ---------
        Q1 — High TES, Low cost  : Deploy freely
        Q2 — High TES, High cost : Prefer on-premises
        Q3 — Low TES,  Low cost  : Acceptable (cost is low; marginal benefit)
        Q4 — Low TES,  High cost : Avoid — poor efficiency at high price

    The cost threshold uses the dataset median rather than a fixed dollar
    value so that the quadrant boundary is data-driven and does not require
    an arbitrary absolute threshold.
    """
    df = tes_df.copy()
    cost_threshold = df["cost_r"].median()

    def assign_quadrant(row):
        high_tes  = row["TES"]    >= tes_threshold
        low_cost  = row["cost_r"] <= cost_threshold
        if high_tes and low_cost:
            return "Q1: Deploy freely"
        elif high_tes and not low_cost:
            return "Q2: Prefer on-prem"
        elif not high_tes and low_cost:
            return "Q3: Acceptable"
        else:
            return "Q4: Avoid"

    df["quadrant"]       = df.apply(assign_quadrant, axis=1)
    df["cost_threshold"] = cost_threshold
    df["tes_threshold"]  = tes_threshold

    return df[["benchmark","model_r","family","TES","cost_r",
               "acc_i_pct","acc_r_pct","delta_acc_pp",
               "quadrant","cost_threshold","tes_threshold"]]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — FIGURE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def fig2_tes_distribution(tes_df: pd.DataFrame, summary_df: pd.DataFrame):
    """
    Figure 2 — TES distribution across benchmarks.

    A grouped bar + individual point overlay, ordered by mean TES descending.
    A horizontal reference line at TES=1 separates strong from marginal zones.
    The IFBench Grok outlier (TES≈30) is shown as an individual annotated
    point above the IFBench bar to preserve the scale of the remaining data.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Exclude Grok outlier from bar/point data for scale, plot separately
    GROK_OUTLIER = "Grok 4.20 0309"
    grok_row = tes_df[
        (tes_df["benchmark"] == "IFBench") &
        (tes_df["model_r"] == GROK_OUTLIER)
    ]
    plot_df = tes_df[tes_df["model_r"] != GROK_OUTLIER]

    benches_ordered = [b for b in BENCH_ORDER if b in plot_df["benchmark"].unique()]
    x_positions = {bench: i for i, bench in enumerate(benches_ordered)}

    bar_width = 0.5

    for bench, xi in x_positions.items():
        colour = BENCH_COLOURS.get(bench, "#888888")
        sub = plot_df[plot_df["benchmark"] == bench]
        summ = summary_df[summary_df["benchmark"] == bench]

        if summ.empty:
            continue

        mean_tes = summ.iloc[0]["mean_TES"]
        std_tes  = summ.iloc[0]["std_TES"]

        # Mean bar
        ax.bar(xi, mean_tes, width=bar_width, color=colour,
               alpha=0.75, label=bench, zorder=2)

        # Std dev error bar
        ax.errorbar(xi, mean_tes, yerr=std_tes, fmt="none",
                    ecolor="black", capsize=5, linewidth=1.5, zorder=3)

        # Individual TES points
        jitter = np.random.default_rng(42).uniform(
            -bar_width * 0.35, bar_width * 0.35, size=len(sub)
        )
        ax.scatter(xi + jitter, sub["TES"], color=colour,
                   edgecolors="white", s=40, zorder=4, linewidths=0.5)

    # Grok outlier annotation
    if not grok_row.empty:
        xi_if = x_positions.get("IFBench", None)
        if xi_if is not None:
            grok_tes = grok_row.iloc[0]["TES"]
            ax.annotate(
                f"Grok 4.20\n(TES={grok_tes:.1f}, outlier\nexcluded from stats)",
                xy=(xi_if, 7.5),
                xytext=(xi_if + 1.0, 8.5),
                arrowprops=dict(arrowstyle="->", color="grey"),
                fontsize=8, color="grey",
            )

    # Reference line at TES=1
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2,
               label="TES = 1 (strong threshold)", zorder=1)

    # Shaded zones
    ax.axhspan(1.0, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 12,
               alpha=0.04, color="green", label="Strong zone (TES > 1)")
    ax.axhspan(ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -0.5, 1.0,
               alpha=0.04, color="orange", label="Marginal zone (TES ≤ 1)")

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(list(x_positions.keys()), rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Token Economy Score (TES)", fontsize=11)
    ax.set_title("Figure 2 — TES Distribution Across Benchmarks\n"
                 "(bars = mean ± std; points = individual model pairs)",
                 fontsize=12)
    ax.set_ylim(-1, 13)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure2_tes_distribution.svg")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def fig3_diminishing_returns(tes_df: pd.DataFrame):
    """
    Figure 3 — Diminishing returns within reasoning effort levels.

    Multi-panel line+point chart.  Each panel shows one model family on one
    benchmark.  X-axis: reasoning effort level ordered by gen_tok_r ascending.
    Left Y-axis: accuracy (%).  Right Y-axis: TES value.

    Cases shown: the six most illustrative from the diminishing returns table.
    """
    CASES = [
        ("GPT",      "GPQA Diamond", "gpt-5.5"),
        ("GPT",      "LiveCodeBench",     "gpt-5.2"),
        ("GPT",      "HLE",          "gpt-5.5"),
        ("DeepSeek", "GPQA Diamond", "v4-pro"),
        ("GPT",      "AIME 2025",    "gpt-5.2"),
        ("DeepSeek", "HLE",          "deepseek-v4-pro"),

    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax_idx, (fam, bench, pid) in enumerate(CASES):
        ax = axes[ax_idx]
        ax2 = ax.twinx()

        # Instruct baseline point
        # Find the instruct row from tes_df (acc_i is constant within a pair)
        sub = tes_df[
            (tes_df["benchmark"] == bench) &
            (tes_df["pair_id"] == pid)
        ].sort_values("gen_tok_r")

        if sub.empty:
            ax.set_visible(False)
            continue

        # Instruct baseline is the shared acc_i_pct + gen_tok_i
        acc_instruct  = sub.iloc[0]["acc_i_pct"]
        gtok_instruct = sub.iloc[0]["gen_tok_i"]

        # Build x/y arrays: instruct + each reasoning variant
        x_gtok = [gtok_instruct] + sub["gen_tok_r"].tolist()
        y_acc  = [acc_instruct]  + sub["acc_r_pct"].tolist()
        y_tes  = [np.nan]        + sub["TES"].tolist()   # no TES for instruct

        colour = BENCH_COLOURS.get(bench, "#888888")

        # Accuracy line (left axis)
        ax.plot(x_gtok, y_acc, "o-", color=colour, linewidth=1.8,
                markersize=7, zorder=3)
        ax.scatter([x_gtok[0]], [y_acc[0]], marker="s", color="grey",
                   s=60, zorder=4, label="Instruct baseline")

        # Annotate instruct point
        ax.annotate("instruct", (x_gtok[0], y_acc[0]),
                    textcoords="offset points", xytext=(6, -10),
                    fontsize=7, color="grey")

        # TES line (right axis, dashed)
        tes_x = sub["gen_tok_r"].tolist()
        tes_y = sub["TES"].tolist()
        ax2.plot(tes_x, tes_y, "^--", color="black", linewidth=1.2,
                 markersize=6, alpha=0.6, label="TES")
        ax2.axhline(1.0, color="black", linestyle=":", linewidth=0.8,
                    alpha=0.4)

        # Labels and formatting
        ax.set_title(f"{fam} · {bench}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Generated tokens", fontsize=8)
        ax.set_ylabel("Accuracy (%)", fontsize=8, color=colour)
        ax2.set_ylabel("TES", fontsize=8, color="black")
        ax.tick_params(axis="both", labelsize=7)
        ax2.tick_params(axis="y", labelsize=7)

        # Highlight negative marginal accuracy (DeepSeek GPQA case)
        for k in range(1, len(y_acc) - 1):
            if y_acc[k + 1] < y_acc[k]:
                ax.annotate("↓ accuracy drops",
                            xy=(x_gtok[k + 1], y_acc[k + 1]),
                            xytext=(0, -18),
                            textcoords="offset points",
                            fontsize=7, color="red",
                            arrowprops=dict(arrowstyle="->", color="red",
                                            lw=0.8))

    fig.suptitle("Figure 3 — Diminishing Returns Within Reasoning Effort Levels\n"
                 "(● accuracy left axis; ▲ TES right axis; □ instruct baseline)",
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure3_diminishing_returns.svg")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def fig4_rcs_stacked(rcs_df: pd.DataFrame, df: pd.DataFrame):
    """
    Figure 4 — Reasoning Cost Share stacked bar chart.

    Shows cost decomposition (input / reasoning / output) for representative
    reasoning model runs.  Bars are sorted by RCS descending.
    Selected models span a range of RCS values for contrast.
    """
    # Select representative rows (one per model-benchmark combination)
    SELECT = [
        ("Claude Sonnet 4.6", "HLE"),
        ("DeepSeek V4 Pro (Max)", "HLE"),
        ("Gemini 3 Flash", "MMLU-Pro"),
        ("GPT 5.2 medium", "MMLU-Pro"),
        ("Claude Opus 4.5", "MMLU-Pro"),
        ("GLM-4.7", "AIME 2025"),
        ("GPT-5.2 (medium)", "AIME 2025"),
        ("Gemini 3 Pro (low)","MMLU-Pro"),
    ]

    rows = []
    for model_name, bench in SELECT:
        hit = df[
            (df["model"] == model_name) &
            (df["benchmark"] == bench) &
            (df["thinking"] == 1)
        ]
        if not hit.empty:
            rows.append(hit.iloc[0])

    if not rows:
        print("  fig4: no matching rows found — skipping")
        return

    plot_df = pd.DataFrame(rows).copy()
    plot_df["RCS"] = plot_df["cost_reasoning"] / plot_df["cost_total"] * 100
    plot_df = plot_df.sort_values("RCS", ascending=True).reset_index(drop=True)
    plot_df["label"] = plot_df["model"] + "\n(" + plot_df["benchmark"] + ")"

    fig, ax = plt.subplots(figsize=(13, 5))

    bar_width = 0.6
    for i, (_, row) in enumerate(plot_df.iterrows()):
        total = row["cost_total"]
        frac_in  = row["cost_input"]     / total
        frac_r   = row["cost_reasoning"] / total
        frac_out = row["cost_output"]    / total

        ax.barh(i, frac_in,  bar_width, color="#CBD5E1", label="Input"     if i == 0 else "")
        ax.barh(i, frac_r,   bar_width, left=frac_in,
                color="#F97316", label="Reasoning" if i == 0 else "")
        ax.barh(i, frac_out, bar_width, left=frac_in + frac_r,
                color="#22C55E", label="Output"    if i == 0 else "")

        # RCS annotation
        ax.text(1.01, i, f"RCS={row['RCS']:.1f}%",
                va="center", fontsize=8, color="#F97316")

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["label"].tolist(), fontsize=8)
    ax.set_xlabel("Fraction of total inference cost", fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.axvline(0.947, color="grey", linestyle="--", linewidth=0.8,
               label="Median RCS")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Figure 4 — Reasoning Cost Share (RCS): Cost Composition per Inference Run\n"
                 "(sorted by RCS descending; right annotations show RCS %)",
                 fontsize=11)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure4_rcs_stacked.svg")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def fig5_quadrant(tes_df: pd.DataFrame):
    """
    Figure 5 — TES vs cloud inference cost quadrant plot.

    X-axis : log-scale cloud inference cost per benchmark run
    Y-axis : TES (linear)

    The four quadrants are separated by:
        TES = 1.0          (horizontal line)
        cost = median cost (vertical line)

    Points are coloured by benchmark.
    Selected representative points are annotated.
    """
    cost_threshold = tes_df["cost_r"].median()

    fig, ax = plt.subplots(figsize=(13, 7))

    for bench in BENCH_ORDER:
        sub = tes_df[tes_df["benchmark"] == bench]
        if sub.empty:
            continue
        colour = BENCH_COLOURS.get(bench, "#888888")
        ax.scatter(sub["cost_r"], sub["TES"],
                   color=colour, s=55, alpha=0.85,
                   edgecolors="white", linewidths=0.5,
                   label=bench, zorder=3)

    # Reference lines
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, zorder=2)
    ax.axvline(cost_threshold, color="black", linestyle="--",
               linewidth=1.2, zorder=2)

    # Quadrant labels
    ax.text(cost_threshold * 0.15, 4,   "Q1\nDeploy freely",   ha="center", va="top", fontsize=9, color="#16A34A", fontweight="bold")
    ax.text(cost_threshold * 6,   4,    "Q2\nPrefer on-prem",  ha="center", va="top", fontsize=9, color="#2563EB", fontweight="bold")
    ax.text(cost_threshold * 0.15, 0.2, "Q3\nAcceptable",      ha="center", va="bottom", fontsize=9, color="#D97706", fontweight="bold")
    ax.text(cost_threshold * 6,   0.2,  "Q4\nAvoid",           ha="center", va="bottom", fontsize=9, color="#DC2626", fontweight="bold")

    # Annotate a few representative points
    ANNOTATE = [
        ("GLM-4.7",          "AIME 2025",    "GLM-4.7\nAIME (TES=11.65)"),
        ("Claude Opus 4.5",  "MMLU-Pro",     "Claude Opus 4.5\nMMLU-Pro (TES=0.16)"),
        ("Qwen3.5 397B A17B","HLE",          "Qwen3.5-397B\nHLE (TES=2.27)"),
        ("GPT-5.5 (xhigh)",  "HLE",          "GPT-5.5 xhigh\nHLE (TES=0.67)"),
    ]
    for model_name, bench, label in ANNOTATE:
        hit = tes_df[
            (tes_df["model_r"] == model_name) &
            (tes_df["benchmark"] == bench)
        ]
        if not hit.empty:
            row = hit.iloc[0]
            ax.annotate(
                label,
                xy=(row["cost_r"], row["TES"]),
                xytext=(row["cost_r"] * 1.6, row["TES"] + 0.6),
                fontsize=7,
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.7),
                color="grey",
            )

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.5)
    ax.set_yticks([-1, 0, 1, 2, 4, 6, 8, 12, 20, 30])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(-1.5, 35)
    ax.set_xlabel("Cloud inference cost per benchmark run (USD, log scale)",
                  fontsize=10)
    ax.set_ylabel("Token Economy Score (TES)", fontsize=10)
    ax.set_title("Figure 5 — TES vs Cloud Inference Cost: Deployment Quadrant Analysis\n"
                 f"(vertical dashed line = dataset median cost ${cost_threshold:.2f}; "
                 "horizontal dashed line = TES threshold 1.0)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure5_quadrant.svg")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — VALIDATION CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def validation_checks(df: pd.DataFrame) -> None:
    """
    Run a suite of sanity checks on the raw data and print a pass/fail report.

    Checks
    ------
    1. Non-reasoning rows must have reasoning_tokens == 0
    2. total_tokens must equal input_tokens + reasoning_tokens + output_tokens
    3. cost_input  = (input_tokens     / 1e6) * price_per_million_input
       cost_reasoning = (reasoning_tokens / 1e6) * price_per_million_output
       cost_output    = (output_tokens    / 1e6) * price_per_million_output
       cost_total     = sum of above  (tolerance $0.01)
    4. All scores in [0, 1]
    5. No duplicate model + benchmark + thinking combinations
    """
    print("\n── Validation Checks ─────────────────────────────────")
    ok = True

    # Check 1
    bad = df[(df["thinking"] == 0) & (df["reasoning_tokens"] > 0)]
    if bad.empty:
        print("  ✅ Check 1: All thinking=0 rows have reasoning_tokens=0")
    else:
        print(f"  ❌ Check 1: {len(bad)} instruct rows have non-zero reasoning_tokens")
        print(bad[["model", "benchmark", "reasoning_tokens"]].to_string(index=False))
        ok = False

    # Check 2
    df["_computed_total"] = (
        df["input_tokens"] + df["reasoning_tokens"] + df["output_tokens"]
    )
    bad2 = df[abs(df["_computed_total"] - df["total_tokens"]) > 100]
    if bad2.empty:
        print("  ✅ Check 2: All total_tokens match component sum (within 100 tokens)")
    else:
        print(f"  ❌ Check 2: {len(bad2)} rows have total_tokens mismatch")
        ok = False

    # Check 3
    tol = 0.02
    df["_exp_ci"]  = df["input_tokens"]     / 1e6 * df["price_per_million_input"]
    df["_exp_cr"]  = df["reasoning_tokens"] / 1e6 * df["price_per_million_output"]
    df["_exp_co"]  = df["output_tokens"]    / 1e6 * df["price_per_million_output"]
    df["_exp_ct"]  = df["_exp_ci"] + df["_exp_cr"] + df["_exp_co"]
    bad3 = df[abs(df["_exp_ct"] - df["cost_total"]) > tol]
    if bad3.empty:
        print(f"  ✅ Check 3: All cost fields match recomputed values (tolerance ${tol})")
    else:
        print(f"  ❌ Check 3: {len(bad3)} rows have cost mismatches > ${tol}")
        print(bad3[["model","benchmark","thinking","cost_total","_exp_ct"]].to_string(index=False))
        ok = False

    # Check 4
    bad4 = df[(df["score"] < 0) | (df["score"] > 1)]
    if bad4.empty:
        print("  ✅ Check 4: All scores in [0, 1]")
    else:
        print(f"  ❌ Check 4: {len(bad4)} scores outside [0, 1]")
        ok = False

    # Check 5
    dupes = df[df.duplicated(subset=["model", "benchmark", "thinking"], keep=False)]
    if dupes.empty:
        print("  ✅ Check 5: No duplicate model+benchmark+thinking rows")
    else:
        print(f"  ⚠️  Check 5: {len(dupes)} duplicate rows found")
        print(dupes[["model","benchmark","thinking"]].to_string(index=False))

    # Clean up temp columns
    for col in ["_computed_total","_exp_ci","_exp_cr","_exp_co","_exp_ct"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    print("── End Validation ────────────────────────────────────\n")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute TES paper metrics from evaluation CSV."
    )
    parser.add_argument(
        "--csv",
        default="actuals_final.csv",
        help="Path to the evaluation CSV (default: actuals_final.csv)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation (useful for quick metric checks)",
    )
    # ADD [] INSIDE THE ARGS () IF WE NEED TO RUN THIS IN COLAB 
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  TES Analysis — {args.csv}")
    print(f"{'='*60}\n")

    # ── Load ─────────────────────────────────────────────────────────────────
    print("Loading data...")
    df = load_data(args.csv)
    print(f"  {len(df)} rows loaded from {args.csv}")

    # ── Validate ─────────────────────────────────────────────────────────────
    validation_checks(df)

    # ── TES pairs ────────────────────────────────────────────────────────────
    print("Computing TES pairs (TES-Δ)...")
    tes_df = calculate_tes_pairs(df)
    print(f"  {len(tes_df)} paired TES values computed")
    tes_df.to_csv(os.path.join(OUTPUT_DIR, "tes_pairs.csv"), index=False)
    print(f"  Saved output/tes_pairs.csv")

    # ── Benchmark summary ────────────────────────────────────────────────────
    print("\nComputing benchmark summary (mean ± std TES)...")
    summary_df = benchmark_summary(tes_df)
    print(summary_df[["benchmark","mean_TES","std_TES","n_pairs_stats",
                       "n_pairs_total","outlier_excluded"]].to_string(index=False))
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "benchmark_summary.csv"), index=False)

    # ── RCS ──────────────────────────────────────────────────────────────────
    print("\nComputing Reasoning Cost Share (RCS)...")
    rcs_df = calculate_rcs(df)
    print(f"  {len(rcs_df)} reasoning runs with RCS computed")
    print(f"  Median RCS: {rcs_df['RCS_pct'].median():.1f}%")
    print(f"  Max RCS:    {rcs_df['RCS_pct'].max():.1f}%  "
          f"({rcs_df.loc[rcs_df['RCS_pct'].idxmax(), 'model']} on "
          f"{rcs_df.loc[rcs_df['RCS_pct'].idxmax(), 'benchmark']})")
    print(f"  Min RCS:    {rcs_df['RCS_pct'].min():.1f}%  "
          f"({rcs_df.loc[rcs_df['RCS_pct'].idxmin(), 'model']} on "
          f"{rcs_df.loc[rcs_df['RCS_pct'].idxmin(), 'benchmark']})")
    rcs_df.to_csv(os.path.join(OUTPUT_DIR, "rcs_table.csv"), index=False)

    # ── DCM ──────────────────────────────────────────────────────────────────
    print("\nComputing Deployment Cost Multiplier (DCM)...")
    dcm_df = calculate_dcm(df)
    print(f"  {len(dcm_df)} on-prem rows with DCM computed")
    dcm_df.to_csv(os.path.join(OUTPUT_DIR, "dcm_table.csv"), index=False)

    print("\n  DCM model summary:")
    dcm_summ = dcm_model_summary(dcm_df)
    print(dcm_summ.to_string(index=False))
    dcm_summ.to_csv(os.path.join(OUTPUT_DIR, "dcm_model_summary.csv"), index=False)

    # ── Diminishing returns ──────────────────────────────────────────────────
    print("\nComputing diminishing returns analysis...")
    dim_df = diminishing_returns(tes_df)
    print(f"  {len(dim_df)} effort-level transitions analysed")
    print(f"  Negative marginal TES (accuracy drops): "
          f"{dim_df['accuracy_drops'].sum()} cases")
    dim_df.to_csv(os.path.join(OUTPUT_DIR, "diminishing_returns.csv"), index=False)

    # ── TES-A (Gemini) ───────────────────────────────────────────────────────
    print("\nComputing TES-A for unpaired models (Gemini)...")
    tes_a_df = calculate_tes_a(df, tes_df)
    if not tes_a_df.empty:
        print(tes_a_df[["benchmark","model_r","model_i",
                         "delta_acc_pp","gen_tok_mult","TES_A"]].to_string(index=False))
        tes_a_df.to_csv(os.path.join(OUTPUT_DIR, "tes_a_gemini.csv"), index=False)
    else:
        print("  No unpaired reasoning models found.")

    # ── Per-question tokens ──────────────────────────────────────────────────
    print("\nComputing per-question token averages...")
    pq_df = per_question_tokens(df)
    pq_df.to_csv(os.path.join(OUTPUT_DIR, "per_question_tokens.csv"), index=False)
    print("  Key rows for Section 7.1 (context window analysis):")
    highlight = pq_df[
        (pq_df["benchmark"] == "HLE") &
        (pq_df["model"].isin([
            "Claude Sonnet 4.6", "GPT-5.5 (xhigh)", "GPT-5.5 (high)"
        ]))
    ]
    if not highlight.empty:
        print(highlight[["model","thinking","reasoning_tok_per_q",
                          "output_tok_per_q","gen_tok_per_q",
                          "context_steps_200k"]].to_string(index=False))

    # ── Quadrant analysis ────────────────────────────────────────────────────
    print("\nRunning quadrant analysis...")
    quad_df = quadrant_analysis(tes_df)
    cost_thr = quad_df["cost_threshold"].iloc[0]
    print(f"  Median cost threshold: ${cost_thr:.2f}")
    counts = quad_df["quadrant"].value_counts()
    print(counts.to_string())
    quad_df.to_csv(os.path.join(OUTPUT_DIR, "quadrant_data.csv"), index=False)

    # ── Figures ──────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\nGenerating figures...")
        fig2_tes_distribution(tes_df, summary_df)
        fig3_diminishing_returns(tes_df)
        fig4_rcs_stacked(rcs_df, df)
        fig5_quadrant(tes_df)
        print("  All figures saved to output/")

    print(f"\n{'='*60}")
    print("  Analysis complete.  All outputs in ./output/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
