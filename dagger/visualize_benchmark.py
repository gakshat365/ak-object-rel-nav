#!/usr/bin/env python3
"""
Benchmark Results Visualizer & Contingency Matrix Generator.
Reads evaluation CSVs (flag_on vs flag_off, or wp_10 vs wp_7) and outputs formatted markdown tables and matrices.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def generate_report(csv_path: str, save_artifact: bool = True) -> str:
    path = Path(csv_path)
    if not path.exists():
        return f"Error: CSV file not found at {csv_path}"

    df = pd.read_csv(path)
    total = len(df)
    if total == 0:
        return "Error: CSV is empty"

    if len(df.columns) < 3:
        return f"Error: CSV must have at least 3 columns (scene, mode_a, mode_b), found: {list(df.columns)}"

    col_a = df.columns[1]
    col_b = df.columns[2]

    name_a = col_a.replace("_", " ").title()
    name_b = col_b.replace("_", " ").title()

    # Outcome counts
    counts_a = df[col_a].value_counts()
    counts_b = df[col_b].value_counts()

    succ_a = counts_a.get('success', 0)
    trap_a = counts_a.get('trapped', 0)
    out_a = counts_a.get('ran out of steps', 0)

    succ_b = counts_b.get('success', 0)
    trap_b = counts_b.get('trapped', 0)
    out_b = counts_b.get('ran out of steps', 0)

    # Contingency cross-tabulation
    ct = pd.crosstab(df[col_a], df[col_b], margins=True)

    # Scene breakdown
    df['scene_id'] = df['scene'].apply(lambda x: x.split('_')[0])
    scene_groups = df.groupby('scene_id')

    lines = []
    lines.append(f"# Benchmark Evaluation Report: {name_a} vs. {name_b}")
    lines.append(f"**Dataset Source**: `{csv_path}`  ")
    lines.append(f"**Total Evaluated Episodes**: **{total}**\n")

    lines.append("## 1. Global Performance Overview\n")
    lines.append(f"| Metric | {name_a} | {name_b} | Delta ({name_a} - {name_b}) |")
    lines.append("|---|:---:|:---:|:---:|")
    lines.append(f"| **Success Rate** | **{succ_a}/{total} ({succ_a/total*100:.1f}%)** | **{succ_b}/{total} ({succ_b/total*100:.1f}%)** | {(succ_a - succ_b)/total*100:+.1f}% |")
    lines.append(f"| **Trapped Rate** | **{trap_a}/{total} ({trap_a/total*100:.1f}%)** | **{trap_b}/{total} ({trap_b/total*100:.1f}%)** | {(trap_a - trap_b)/total*100:+.1f}% |")
    lines.append(f"| **Ran Out of Steps (Survivals)** | **{out_a}/{total} ({out_a/total*100:.1f}%)** | **{out_b}/{total} ({out_b/total*100:.1f}%)** | {(out_a - out_b)/total*100:+.1f}% |\n")

    lines.append("## 2. Contingency Matrix (Paired Outcomes)\n")
    lines.append(f"| {name_a} \\ {name_b} | Success | Trapped | Ran Out of Steps | Total {name_a} |")
    lines.append("|---|:---:|:---:|:---:|:---:|")

    for label in ['success', 'trapped', 'ran out of steps']:
        if label in ct.index:
            s = ct.loc[label].get('success', 0)
            t = ct.loc[label].get('trapped', 0)
            r = ct.loc[label].get('ran out of steps', 0)
            tot = ct.loc[label].get('All', s + t + r)
            lines.append(f"| **{label.title()}** | **{s}** | **{t}** | **{r}** | **{tot}** |")

    # Margins row
    s_all = ct['success'].get('All', succ_b) if 'success' in ct else 0
    t_all = ct['trapped'].get('All', trap_b) if 'trapped' in ct else 0
    r_all = ct['ran out of steps'].get('All', out_b) if 'ran out of steps' in ct else 0
    lines.append(f"| **Total {name_b}** | **{s_all}** | **{t_all}** | **{r_all}** | **{total}** |\n")

    # Key head-to-head metrics
    a_wins = len(df[(df[col_a] == 'success') & (df[col_b] != 'success')])
    b_wins = len(df[(df[col_b] == 'success') & (df[col_a] != 'success')])
    both_succ = len(df[(df[col_a] == 'success') & (df[col_b] == 'success')])
    both_trap = len(df[(df[col_a] == 'trapped') & (df[col_b] == 'trapped')])

    lines.append("### Key Head-to-Head Statistics")
    lines.append(f"- **Both Succeeded**: **{both_succ}** episodes ({both_succ/total*100:.1f}%)")
    lines.append(f"- **{name_a} Unique Wins** ({name_a} succeeded, {name_b} failed): **{a_wins}** episodes")
    lines.append(f"- **{name_b} Unique Wins** ({name_b} succeeded, {name_a} failed): **{b_wins}** episodes")
    lines.append(f"- **Both Trapped**: **{both_trap}** episodes ({both_trap/total*100:.1f}%)\n")

    lines.append("## 3. Performance Breakdown by Scene\n")
    lines.append(f"| Scene ID | Total Episodes | {name_a} Success (%) | {name_b} Success (%) | {name_a} Trapped | {name_b} Trapped | Advantage |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    for sc, grp in scene_groups:
        n_sc = len(grp)
        sc_succ_a = len(grp[grp[col_a] == 'success'])
        sc_succ_b = len(grp[grp[col_b] == 'success'])
        sc_trap_a = len(grp[grp[col_a] == 'trapped'])
        sc_trap_b = len(grp[grp[col_b] == 'trapped'])

        if sc_succ_a > sc_succ_b:
            adv = f"**{name_a} (+{sc_succ_a - sc_succ_b})**"
        elif sc_succ_b > sc_succ_a:
            adv = f"**{name_b} (+{sc_succ_b - sc_succ_a})**"
        else:
            adv = "Tied"

        lines.append(f"| `{sc}` | {n_sc} | {sc_succ_a} ({sc_succ_a/n_sc*100:.1f}%) | {sc_succ_b} ({sc_succ_b/n_sc*100:.1f}%) | {sc_trap_a} | {sc_trap_b} | {adv} |")

    report_text = "\n".join(lines)
    return report_text

def main():
    parser = argparse.ArgumentParser(description="Visualize Benchmark Results")
    parser.add_argument("--csv", type=str, default="dagger/benchmark_results.csv", help="Path to results CSV")
    parser.add_argument("--out", type=str, default=None, help="Optional output markdown file")
    args = parser.parse_args()

    report = generate_report(args.csv)
    print(report)

    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"\nReport written to {args.out}")

if __name__ == "__main__":
    main()
