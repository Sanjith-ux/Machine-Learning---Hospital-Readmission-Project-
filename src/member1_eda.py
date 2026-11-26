"""Generate five aggregate figures without modifying raw or processed datasets."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_quality_validation import ROOT, load_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/diabetic_data.csv")
    parser.add_argument("--processed", type=Path, default=ROOT / "results/processed_data.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/eda")
    args = parser.parse_args()
    raw, cleaned = load_inputs(args.raw, args.processed)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    def save(name, title, xlabel, ylabel):
        plt.title(title, pad=16)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(args.output / name, dpi=160)
        plt.close()

    literal_missing = (raw.isin(["", "?"]) | raw.isna()).mean() * 100
    shared_missing = pd.read_csv(args.raw).replace("?", pd.NA).isna().mean() * 100
    summary = pd.DataFrame({"Unknown source tokens": literal_missing,
                            "Shared parser missing": shared_missing})
    summary = summary.loc[summary.max(axis=1).gt(0)].sort_values("Shared parser missing")
    summary.plot.barh(figsize=(10, 6), color=["#21918c", "#d98b36"])
    plt.xlim(0, 105)
    save("missing_value_percentages.png", "Original cohort: missingness depends on parsing\nLiteral None is a not-performed category", "Encounters (%)", "Feature")

    target = cleaned["target"].value_counts().reindex(["0", "1"], fill_value=0)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(["0: NO or >30", "1: <30"], target, color=["#31688e", "#d98b36"])
    for bar, count in zip(bars, target):
        plt.text(bar.get_x()+bar.get_width()/2, count, f"{count:,}\n{count/len(cleaned):.2%}", ha="center", va="bottom")
    plt.ylim(0, max(target) * 1.2)
    save("target_distribution.png", "Cleaned cohort: readmission target", "Target", "Encounters")

    ages = [f"[{n}-{n+10})" for n in range(0, 100, 10)]
    pd.DataFrame({"Original": raw["age"].value_counts(), "Cleaned": cleaned["age"].value_counts()}).reindex(ages, fill_value=0).plot.bar(figsize=(11, 5), color=["#31688e", "#21918c"], rot=35)
    save("age_distribution.png", "Encounter counts by age band", "Age band (years)", "Encounters (not unique patients)")

    plt.figure(figsize=(8, 5))
    counts = [len(raw), len(cleaned)]
    bars = plt.bar(["Original", "Cleaned"], counts, color=["#31688e", "#21918c"])
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x()+bar.get_width()/2, count, f"{count:,}", ha="center", va="bottom")
    plt.ylim(0, max(counts) * 1.15)
    save("encounter_counts.png", f"Cohort cleaning: {len(raw)-len(cleaned):,} encounters excluded", "Cohort", "Encounters")

    distribution = raw["readmitted"].value_counts().reindex(["NO", ">30", "<30"], fill_value=0)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(distribution.index, distribution, color=["#31688e", "#21918c", "#d98b36"])
    for bar, count in zip(bars, distribution):
        plt.text(bar.get_x()+bar.get_width()/2, count, f"{count:,}\n{count/len(raw):.2%}", ha="center", va="bottom")
    plt.ylim(0, max(distribution) * 1.2)
    save("original_readmission_distribution.png", "Original cohort: recorded readmission categories", "Readmitted", "Encounters")
    print(f"Five aggregate figures saved in {args.output}")


if __name__ == "__main__":
    main()
