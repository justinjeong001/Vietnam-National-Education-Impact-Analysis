"""
================================================================================
TOPIK National Accreditation — Module 1: Demographic & Descriptive Analysis
Vietnamese Ministry of Education & Training (MoET) × MoE Korea
================================================================================
Author   : Senior Policy Consultant & Data Architect
Dataset  : TOPIK Strategic Pilot, N=5,114 (Vietnam)

This is Module 1 of the two-script pipeline.
Run this FIRST to generate plots 01–04 (demographic/descriptive layer).
Then run topik_policy_case.py to generate plots 05–07 (analytical layer).

Outputs
-------
  01_respondent_profile.png     — Gender & age distribution of the pilot cohort
  02_learning_background.png    — Study duration breakdown + TOPIK experience rate
  03_score_overview.png         — Mean rating per exam dimension (all 8 items)
  04_topik_purpose_breakdown.png — Why Vietnamese students are taking TOPIK

Usage
-----
  # Default: CSV beside this script, outputs/ beside this script
  python topik_analysis.py

  # Explicit paths
  python topik_analysis.py --data path/to/topik_survey_final.csv --out path/to/outputs/

Dependencies: See requirements.txt
Python       : 3.9+
================================================================================
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path
from collections import Counter

# ── Path resolution ────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TOPIK Module 1 — Demographic & Descriptive Analysis"
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=_SCRIPT_DIR / "topik_survey_final.csv",
        help="Path to survey CSV (default: topik_survey_final.csv beside this script)",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=_SCRIPT_DIR / "outputs",
        help="Output directory for PNGs (default: outputs/ beside this script)",
    )
    return parser.parse_args()

# ── Design System (mirrors topik_policy_case.py for visual consistency) ────────
C_NAVY   = "#1B3A6B"
C_RED    = "#C0392B"
C_GOLD   = "#D4A017"
C_GREEN  = "#1A7A4A"
C_LIGHT  = "#F4F7FB"
C_GREY   = "#7F8C8D"
C_ORANGE = "#E67E22"
C_TEAL   = "#16A085"

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
    "axes.titleweight" : "bold",
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 10,
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
})

FOOTER = "Ministry of Education, Republic of Korea × MoET Vietnam  |  TOPIK Policy Analysis 2025–26"

# ── Column names (post-strip) ──────────────────────────────────────────────────
SCORE_COLS = [
    "listening_appropriateness_score",
    "reading_appropriateness_score",
    "writing_appropriateness_score",
    "speaking_assessment_score",
    "exam_time_sufficiency",
    "question_count_rating",
    "question_diversity_rating",
    "instruction_clarity_score",
]
SCORE_LABELS = {
    "listening_appropriateness_score" : "Listening\nAppropriateness",
    "reading_appropriateness_score"   : "Reading\nAppropriateness",
    "writing_appropriateness_score"   : "Writing\nAppropriateness",
    "speaking_assessment_score"       : "Speaking\nAssessment",
    "exam_time_sufficiency"           : "Time\nSufficiency",
    "question_count_rating"           : "Question\nCount",
    "question_diversity_rating"       : "Question\nDiversity",
    "instruction_clarity_score"       : "Instruction\nClarity",
}

DURATION_ORDER = {
    "Less than 6 months"                        : 1,
    "6 months to 1 year"                        : 2,
    "1 year to less than 1 year and 6 months"   : 3,
    "1 year 6 months to less than 2 years"      : 4,
    "more than 2 years"                         : 5,
}
DURATION_SHORT = {
    "Less than 6 months"                        : "< 6 mo",
    "6 months to 1 year"                        : "6–12 mo",
    "1 year to less than 1 year and 6 months"   : "1–1.5 yr",
    "1 year 6 months to less than 2 years"      : "1.5–2 yr",
    "more than 2 years"                         : "> 2 yr",
}


# ══════════════════════════════════════════════════════════════════════════════
# LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load(path: str) -> pd.DataFrame:
    """
    Load and clean the survey CSV.

    Steps:
      1. Strip all column-name whitespace/tabs.
      2. Drop empty trailing column.
      3. Deduplicate rows (81 exact duplicates in raw file).
      4. Median-impute score columns (36% MNAR missingness — see topik_policy_case.py).
      5. Engineer ordinal duration and short-label columns.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["Unnamed: 21"], errors="ignore")

    # Deduplicate
    n_raw = len(df)
    df = df.drop_duplicates()
    n_removed = n_raw - len(df)
    if n_removed:
        print(f"  [DATA] Removed {n_removed} exact duplicates ({n_raw} → {len(df)} records).")

    # Median impute score columns
    for c in SCORE_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # Convenience renames
    df = df.rename(columns={
        "Please select your gender."                       : "gender",
        "Please enter your current age."                   : "age",
        "How long is the Korean language learning period?" : "learning_duration",
        "Have you ever taken the TOPIK exam?"              : "topik_experience",
    })

    df["duration_ordinal"] = df["learning_duration"].map(DURATION_ORDER)
    df["duration_short"]   = df["learning_duration"].map(DURATION_SHORT)
    return df


def _add_footer(fig, text: str = FOOTER) -> None:
    fig.text(0.5, 0.01, text, ha="center", va="bottom",
             fontsize=7, color=C_GREY, style="italic")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 01 — RESPONDENT PROFILE (gender + age)
# ══════════════════════════════════════════════════════════════════════════════

def plot_respondent_profile(df: pd.DataFrame, out_path: str) -> None:
    """
    Two-panel figure:
      Left  — Gender breakdown (donut chart)
      Right — Age distribution (histogram with KDE overlay)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(C_LIGHT)
    for ax in (ax1, ax2):
        ax.set_facecolor(C_LIGHT)

    # ── Panel A: Gender donut ──────────────────────────────────────────────
    gender_counts = df["gender"].value_counts()
    colors  = [C_NAVY, C_TEAL, C_GOLD]
    wedges, texts, autotexts = ax1.pie(
        gender_counts.values,
        labels=None,
        colors=colors[:len(gender_counts)],
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
    )
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")
        autotext.set_color("white")

    # Legend
    legend_patches = [
        mpatches.Patch(color=colors[i], label=f"{label} (n={v:,})")
        for i, (label, v) in enumerate(gender_counts.items())
    ]
    ax1.legend(handles=legend_patches, loc="lower center",
               bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9,
               frameon=False)
    ax1.set_title("Gender Distribution", pad=16, fontsize=13, color=C_NAVY)
    ax1.text(0, 0, f"N = {len(df):,}", ha="center", va="center",
             fontsize=12, fontweight="bold", color=C_NAVY)

    # ── Panel B: Age histogram + KDE ──────────────────────────────────────
    ages = df["age"].dropna()
    ax2.hist(ages, bins=range(int(ages.min()), int(ages.max()) + 2),
             color=C_NAVY, alpha=0.75, edgecolor="white", linewidth=0.5, density=True)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(ages, bw_method=0.3)
    xs  = np.linspace(ages.min(), ages.max(), 300)
    ax2.plot(xs, kde(xs), color=C_RED, linewidth=2.5, label="KDE")

    # Median line
    med = ages.median()
    ax2.axvline(med, color=C_GOLD, linewidth=1.8, linestyle="--",
                label=f"Median = {med:.0f} yr")
    ax2.set_xlabel("Age (years)", labelpad=6)
    ax2.set_ylabel("Density", labelpad=6)
    ax2.set_title("Age Distribution of Pilot Respondents", pad=12, color=C_NAVY)
    ax2.legend(fontsize=9, frameon=False)

    # Annotation box
    ax2.text(0.97, 0.93,
             f"Mean  {ages.mean():.1f} yr\nStd   {ages.std():.1f} yr\nRange {int(ages.min())}–{int(ages.max())} yr",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=8.5, color=C_NAVY,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor=C_GREY, alpha=0.8))

    fig.suptitle("TOPIK Vietnam Pilot — Respondent Profile", fontsize=15,
                 fontweight="bold", color=C_NAVY, y=1.01)
    _add_footer(fig)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=C_LIGHT)
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 02 — LEARNING BACKGROUND (duration + TOPIK experience)
# ══════════════════════════════════════════════════════════════════════════════

def plot_learning_background(df: pd.DataFrame, out_path: str) -> None:
    """
    Two-panel figure:
      Left  — Korean study duration (ordered horizontal bar)
      Right — TOPIK exam experience rate (stacked bar with percentage labels)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(C_LIGHT)
    for ax in (ax1, ax2):
        ax.set_facecolor(C_LIGHT)

    # ── Panel A: Duration bar chart ────────────────────────────────────────
    dur_df = (
        df.groupby("learning_duration")
          .size()
          .reset_index(name="count")
    )
    dur_df["ordinal"] = dur_df["learning_duration"].map(DURATION_ORDER)
    dur_df["short"]   = dur_df["learning_duration"].map(DURATION_SHORT)
    dur_df = dur_df.sort_values("ordinal")
    dur_df["pct"] = dur_df["count"] / dur_df["count"].sum() * 100

    bar_colors = [C_NAVY, C_TEAL, C_GREEN, C_ORANGE, C_RED]
    bars = ax1.barh(range(len(dur_df)), dur_df["count"], color=bar_colors,
                    edgecolor="white", linewidth=0.8, height=0.65)
    ax1.set_yticks(range(len(dur_df)))
    ax1.set_yticklabels(dur_df["short"], fontsize=10)
    ax1.set_xlabel("Number of Respondents", labelpad=6)
    ax1.set_title("Korean Study Duration", pad=12, color=C_NAVY)

    for i, (bar, row) in enumerate(zip(bars, dur_df.itertuples())):
        ax1.text(bar.get_width() + 18, bar.get_y() + bar.get_height() / 2,
                 f"{row.count:,}  ({row.pct:.1f}%)",
                 va="center", fontsize=8.5, color=C_NAVY)

    ax1.set_xlim(0, dur_df["count"].max() * 1.22)
    ax1.tick_params(axis="x", length=0)

    # Highlight "most experienced" bar
    max_idx = dur_df["count"].idxmax()
    pos = dur_df.index.get_loc(max_idx)
    ax1.annotate("Largest cohort", xy=(dur_df.iloc[pos]["count"], pos),
                 xytext=(dur_df.iloc[pos]["count"] * 0.5, pos + 0.55),
                 fontsize=8, color=C_RED,
                 arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))

    # ── Panel B: TOPIK experience stacked bar ─────────────────────────────
    exp_counts = df["topik_experience"].value_counts()
    labels     = exp_counts.index.tolist()
    # Normalise labels for display
    display_labels = []
    for lbl in labels:
        if "does not" in lbl.lower() or "not exist" in lbl.lower():
            display_labels.append("Never taken TOPIK")
        else:
            display_labels.append("Has taken TOPIK")

    n_total = exp_counts.sum()
    pcts    = exp_counts.values / n_total * 100
    colors  = [C_GREEN if "there" in lbl else C_GREY for lbl in labels]

    bars2 = ax2.bar(["TOPIK Experience"], [pcts[0]], color=colors[0],
                    width=0.4, label=display_labels[0], edgecolor="white")
    ax2.bar(["TOPIK Experience"], [pcts[1]], bottom=[pcts[0]], color=colors[1],
            width=0.4, label=display_labels[1], edgecolor="white")

    # Labels inside bars
    ax2.text(0, pcts[0] / 2, f"{pcts[0]:.1f}%\n({exp_counts.values[0]:,})",
             ha="center", va="center", fontsize=13, fontweight="bold",
             color="white")
    ax2.text(0, pcts[0] + pcts[1] / 2, f"{pcts[1]:.1f}%\n({exp_counts.values[1]:,})",
             ha="center", va="center", fontsize=13, fontweight="bold",
             color="white")

    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Percentage of Respondents (%)", labelpad=6)
    ax2.set_title("TOPIK Examination Experience", pad=12, color=C_NAVY)
    ax2.legend(loc="upper right", fontsize=9, frameon=False)
    ax2.set_xticks([])

    # Policy note
    has_taken_pct = pcts[0] if "there" in labels[0] else pcts[1]
    ax2.text(0.5, -0.12,
             f"{has_taken_pct:.1f}% of respondents have direct TOPIK experience\n"
             f"→ their rating data carries higher validity weight",
             ha="center", transform=ax2.transAxes,
             fontsize=8, color=C_GREY, style="italic")

    fig.suptitle("TOPIK Vietnam Pilot — Learning Background", fontsize=15,
                 fontweight="bold", color=C_NAVY, y=1.01)
    _add_footer(fig)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=C_LIGHT)
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 03 — SCORE OVERVIEW (all 8 exam dimensions)
# ══════════════════════════════════════════════════════════════════════════════

def plot_score_overview(df: pd.DataFrame, out_path: str) -> None:
    """
    Diverging horizontal bar chart showing mean score per exam dimension,
    centred on the neutral midpoint (3.0 on a 5-point Likert scale).
    Color-codes below/above midpoint to surface structural weaknesses immediately.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(C_LIGHT)
    ax.set_facecolor(C_LIGHT)

    means = {col: df[col].mean() for col in SCORE_COLS if col in df.columns}
    labels_short = [SCORE_LABELS[c] for c in means]
    values = list(means.values())
    MIDPOINT = 3.0

    # Sort by score descending
    order  = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    labels_sorted = [labels_short[i] for i in order]
    values_sorted = [values[i] for i in order]

    bar_colors = [C_GREEN if v >= MIDPOINT else C_RED for v in values_sorted]
    bars = ax.barh(range(len(values_sorted)),
                   [v - MIDPOINT for v in values_sorted],
                   left=MIDPOINT,
                   color=bar_colors, alpha=0.82,
                   edgecolor="white", linewidth=0.8, height=0.68)

    # Midpoint line
    ax.axvline(MIDPOINT, color=C_GREY, linewidth=1.2, linestyle="--", alpha=0.7)

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, values_sorted)):
        x_label = val + 0.04 if val >= MIDPOINT else val - 0.04
        ha = "left" if val >= MIDPOINT else "right"
        ax.text(x_label, i, f"{val:.3f}", va="center", ha=ha,
                fontsize=9.5, fontweight="bold", color=C_NAVY)

    ax.set_yticks(range(len(labels_sorted)))
    ax.set_yticklabels(labels_sorted, fontsize=9.5)
    ax.set_xlim(1.8, 4.6)
    ax.set_xlabel("Mean Rating (5-point Likert Scale)", labelpad=6)
    ax.set_title("TOPIK Exam Perception — All 8 Dimensions\n"
                 "(Bars diverge from neutral midpoint = 3.0)",
                 pad=12, color=C_NAVY)

    # Legend
    legend_handles = [
        mpatches.Patch(color=C_GREEN, alpha=0.82, label="Above neutral (≥ 3.0) ✓"),
        mpatches.Patch(color=C_RED,   alpha=0.82, label="Below neutral (< 3.0) ⚠"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=False)

    # Annotation for lowest scorer
    min_idx = values_sorted.index(min(values_sorted))
    ax.annotate("Lowest-rated dimension\n→ prioritise in reform",
                xy=(values_sorted[min_idx], min_idx),
                xytext=(values_sorted[min_idx] - 0.55, min_idx + 1.1),
                fontsize=8, color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))

    _add_footer(fig)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=C_LIGHT)
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 04 — TOPIK PURPOSE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════

def plot_purpose_breakdown(df: pd.DataFrame, out_path: str) -> None:
    """
    Parses the multi-choice purpose column, extracts top atomic motivations,
    and plots a ranked horizontal bar chart with percentage labels.
    Surfaces the policy-critical signal: majority motivation is career/study mobility.
    """
    purpose_col = "What is your purpose for taking the exam? (Multiple choices possible)"
    if purpose_col not in df.columns:
        print(f"  [WARN] Purpose column not found — skipping plot 04.")
        return

    # Parse comma-separated multi-choice responses
    counter: Counter = Counter()
    for row in df[purpose_col].dropna():
        for item in str(row).split(","):
            item = item.strip()
            # Normalise synonymous labels
            item = item.replace("Study in Korea", "Studying in Korea")
            item = item.replace("verifying Korean language proficiency",
                                "Checking Korean language skills")
            item = item.replace("Check your Korean skills",
                                "Checking Korean language skills")
            item = item.replace("Checking Korean Language Skills",
                                "Checking Korean language skills")
            item = item.replace("verifying Korean language skills",
                                "Checking Korean language skills")
            if item:
                counter[item] += 1

    # Top N purposes
    TOP_N = 8
    top_items  = counter.most_common(TOP_N)
    labels_raw = [item[0] for item in top_items]
    counts     = [item[1] for item in top_items]
    total_resp = len(df[purpose_col].dropna())
    pcts       = [c / total_resp * 100 for c in counts]

    # Sort ascending for horizontal bar (bottom = largest)
    order   = sorted(range(TOP_N), key=lambda i: counts[i])
    labels  = [labels_raw[i] for i in order]
    pcts_s  = [pcts[i] for i in order]
    counts_s = [counts[i] for i in order]

    # Colour scale: career/study = navy, skills = teal, culture = gold, other = grey
    def _pick_color(label: str) -> str:
        low = label.lower()
        if "korea" in low and ("study" in low or "admission" in low):
            return C_NAVY
        if "employ" in low:
            return C_RED
        if "skill" in low or "proficiency" in low:
            return C_TEAL
        if "culture" in low or "interest" in low:
            return C_GOLD
        return C_GREY

    bar_colors = [_pick_color(l) for l in labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(C_LIGHT)
    ax.set_facecolor(C_LIGHT)

    bars = ax.barh(range(TOP_N), pcts_s, color=bar_colors, alpha=0.85,
                   edgecolor="white", linewidth=0.8, height=0.68)
    ax.set_yticks(range(TOP_N))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("% of Respondents (multi-choice — totals exceed 100%)", labelpad=6)
    ax.set_title("Why Vietnamese Students Take TOPIK\n"
                 f"(Top {TOP_N} motivations from N={total_resp:,} responses; multi-choice)",
                 pad=12, color=C_NAVY)

    for i, (bar, pct, cnt) in enumerate(zip(bars, pcts_s, counts_s)):
        ax.text(bar.get_width() + 0.4, i,
                f"{pct:.1f}%  (n={cnt:,})",
                va="center", fontsize=8.5, color=C_NAVY)

    ax.set_xlim(0, max(pcts_s) * 1.28)
    ax.tick_params(axis="x", length=0)

    # Legend
    legend_patches = [
        mpatches.Patch(color=C_NAVY, alpha=0.85, label="Study mobility (Korea universities)"),
        mpatches.Patch(color=C_RED,  alpha=0.85, label="Employment / career"),
        mpatches.Patch(color=C_TEAL, alpha=0.85, label="Skills verification"),
        mpatches.Patch(color=C_GOLD, alpha=0.85, label="Cultural interest"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8.5, frameon=False)

    # Policy note
    ax.text(0.5, -0.13,
            "Policy implication: Dominant motivations (study + employment) confirm TOPIK graduation credit\n"
            "directly enables post-secondary mobility — the core MoET policy argument.",
            ha="center", transform=ax.transAxes, fontsize=8, color=C_GREY, style="italic")

    _add_footer(fig)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=C_LIGHT)
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = _parse_args()

    data_path = args.data.resolve()
    out_dir   = args.out.resolve()

    if not data_path.exists():
        raise FileNotFoundError(
            f"Survey CSV not found: {data_path}\n"
            f"Pass the correct path with --data <path>"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║   TOPIK — MODULE 1: DEMOGRAPHIC & DESCRIPTIVE ANALYSIS          ║")
    print("║   Korea MoE × Vietnam MoET  |  January 2026 Decree Support      ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    print(f"  Data  : {data_path}")
    print(f"  Output: {out_dir}\n")

    df = load(str(data_path))
    print(f"  [DATA] Final working dataset: {len(df):,} rows × {df.shape[1]} columns\n")

    print("=" * 70)
    print("  GENERATING PLOTS 01–04")
    print("=" * 70)

    plot_respondent_profile(df,   str(out_dir / "01_respondent_profile.png"))
    plot_learning_background(df,  str(out_dir / "02_learning_background.png"))
    plot_score_overview(df,       str(out_dir / "03_score_overview.png"))
    plot_purpose_breakdown(df,    str(out_dir / "04_topik_purpose_breakdown.png"))

    print()
    print(f"  All outputs → {out_dir}")
    print("  Module 1 complete. Run topik_policy_case.py for plots 05–07.\n")


if __name__ == "__main__":
    main()
