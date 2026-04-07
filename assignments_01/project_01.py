"""World Happiness project pipeline using Prefect"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from prefect import flow, task, get_run_logger

DATA_DIR = Path(__file__).resolve().parents[1] / "assignments" / "resources" / "happiness_project"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: c.strip())
    df = df.rename(
        columns={
            "Ranking": "ranking",
            "Country": "country",
            "Regional indicator": "region",
            "Happiness score": "happiness_score",
            "GDP per capita": "gdp_per_capita",
            "Social support": "social_support",
            "Healthy life expectancy": "healthy_life_expectancy",
            "Freedom to make life choices": "freedom",
            "Generosity": "generosity",
            "Perceptions of corruption": "perceptions_of_corruption",
        }
    )
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df


@task(retries=3, retry_delay_seconds=2)
def load_happiness_data() -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Loading data from {DATA_DIR}")

    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}. Place the happiness CSV files there"
        )

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    frames = []
    for csv_file in csv_files:
        logger.info(f"Reading {csv_file.name}")
        year_match = re.search(r"(\d{4})", csv_file.name)
        if not year_match:
            raise ValueError(f"Could not infer year from file name: {csv_file.name}")
        year = int(year_match.group(1))

        df = pd.read_csv(csv_file, sep=";", decimal=",", encoding="utf-8")
        df = _standardize_df(df)
        df["year"] = year
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(OUTPUT_DIR / "merged_happiness.csv", index=False)
    logger.info("Saved merged dataset to outputs/merged_happiness.csv")
    return merged


@task
def descriptive_statistics(df: pd.DataFrame) -> dict:
    logger = get_run_logger()
    happiness = df["happiness_score"].dropna()
    mean = happiness.mean()
    median = happiness.median()
    std = happiness.std()
    logger.info(f"Overall happiness mean = {mean:.4f}")
    logger.info(f"Overall happiness median = {median:.4f}")
    logger.info(f"Overall happiness std = {std:.4f}")

    mean_by_year = df.groupby("year")["happiness_score"].mean()
    logger.info("Mean happiness by year:")
    for year, value in mean_by_year.items():
        logger.info(f"  {year}: {value:.4f}")

    if "region" in df.columns:
        mean_by_region = df.groupby("region")["happiness_score"].mean().sort_values(ascending=False)
        logger.info("Mean happiness by region:")
        for region, value in mean_by_region.items():
            logger.info(f"  {region}: {value:.4f}")
    else:
        mean_by_region = pd.Series(dtype=float)

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "mean_by_year": mean_by_year,
        "mean_by_region": mean_by_region,
    }


@task
def create_plots(df: pd.DataFrame) -> None:
    logger = get_run_logger()

    histogram_path = OUTPUT_DIR / "happiness_histogram.png"
    plt.figure(figsize=(8, 5))
    sns.histplot(df["happiness_score"].dropna(), bins=20, kde=False, color="skyblue")
    plt.title("Happiness Score Distribution")
    plt.xlabel("Happiness Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(histogram_path)
    plt.close()
    logger.info(f"Saved histogram to {histogram_path}")

    boxplot_path = OUTPUT_DIR / "happiness_by_year.png"
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="year", y="happiness_score", data=df.sort_values("year"), palette="pastel")
    plt.title("Happiness Score by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.tight_layout()
    plt.savefig(boxplot_path)
    plt.close()
    logger.info(f"Saved year-by-year boxplot to {boxplot_path}")

    scatter_path = OUTPUT_DIR / "gdp_vs_happiness.png"
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="gdp_per_capita", y="happiness_score", data=df, alpha=0.7)
    plt.title("GDP per Capita vs Happiness Score")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()
    logger.info(f"Saved GDP vs happiness scatter plot to {scatter_path}")

    heatmap_path = OUTPUT_DIR / "correlation_heatmap.png"
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    logger.info(f"Saved correlation heatmap to {heatmap_path}")


@task
def hypothesis_testing(df: pd.DataFrame) -> dict:
    logger = get_run_logger()
    scores_2019 = df.loc[df["year"] == 2019, "happiness_score"].dropna()
    scores_2020 = df.loc[df["year"] == 2020, "happiness_score"].dropna()

    if scores_2019.empty or scores_2020.empty:
        raise ValueError("Missing 2019 or 2020 happiness score data for hypothesis testing.")

    t_stat, p_value = stats.ttest_ind(scores_2019, scores_2020, equal_var=False)
    mean_2019 = scores_2019.mean()
    mean_2020 = scores_2020.mean()

    logger.info("2019 vs 2020 happiness score comparison")
    logger.info(f"  2019 mean = {mean_2019:.4f}")
    logger.info(f"  2020 mean = {mean_2020:.4f}")
    logger.info(f"  t-statistic = {t_stat:.4f}")
    logger.info(f"  p-value = {p_value:.4f}")

    if p_value < 0.05:
        interpretation = (
            "The difference in average happiness between 2019 and 2020 is statistically significant at alpha = 0.05, "
            "suggesting the pandemic period saw a real change in global happiness scores."
        )
    else:
        interpretation = (
            "The difference in average happiness between 2019 and 2020 is not statistically significant at alpha = 0.05, "
            "so we cannot conclude the pandemic caused a meaningful global shift in happiness from these data alone."
        )
    logger.info(f"  interpretation: {interpretation}")

    region_means = df.groupby("region")["happiness_score"].mean().sort_values(ascending=False)
    top_region = region_means.index[0]
    bottom_region = region_means.index[-1]
    top_scores = df.loc[df["region"] == top_region, "happiness_score"].dropna()
    bottom_scores = df.loc[df["region"] == bottom_region, "happiness_score"].dropna()
    region_t_stat, region_p_value = stats.ttest_ind(top_scores, bottom_scores, equal_var=False)

    logger.info(f"Comparing top region ({top_region}) to bottom region ({bottom_region})")
    logger.info(f"  {top_region} mean = {top_scores.mean():.4f}")
    logger.info(f"  {bottom_region} mean = {bottom_scores.mean():.4f}")
    logger.info(f"  t-statistic = {region_t_stat:.4f}")
    logger.info(f"  p-value = {region_p_value:.4f}")

    return {
        "pre_post_t_test": {
            "t_stat": t_stat,
            "p_value": p_value,
            "mean_2019": mean_2019,
            "mean_2020": mean_2020,
            "interpretation": interpretation,
        },
        "regional_t_test": {
            "top_region": top_region,
            "bottom_region": bottom_region,
            "t_stat": region_t_stat,
            "p_value": region_p_value,
        },
    }


@task
def correlation_analysis(df: pd.DataFrame) -> dict:
    logger = get_run_logger()
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != "happiness_score"
    ]
    logger.info(f"Computing correlations for numeric variables: {numeric_cols}")

    correlations = []
    for col in numeric_cols:
        valid = df[[col, "happiness_score"]].dropna()
        if len(valid) < 2:
            continue
        coef, p_value = pearsonr(valid[col], valid["happiness_score"])
        significant = p_value < 0.05
        correlations.append(
            {
                "variable": col,
                "coef": coef,
                "p_value": p_value,
                "significant": significant,
            }
        )
        logger.info(f"  {col}: coef = {coef:.4f}, p = {p_value:.4f}, significant = {significant}")

    adjusted_alpha = 0.05 / max(len(correlations), 1)
    logger.info(f"Bonferroni-adjusted alpha = {adjusted_alpha:.6f}")

    for result in correlations:
        result["significant_bonferroni"] = result["p_value"] < adjusted_alpha
        logger.info(
            f"  {result['variable']}: significant at 0.05 = {result['significant']}, "
            f"after correction = {result['significant_bonferroni']}"
        )

    return {
        "correlations": correlations,
        "adjusted_alpha": adjusted_alpha,
    }


@task
def summary_report(
    df: pd.DataFrame,
    stats_summary: dict,
    t_test_summary: dict,
    correlation_summary: dict,
) -> None:
    logger = get_run_logger()
    num_countries = df["country"].nunique()
    num_years = df["year"].nunique()
    logger.info(f"Total countries in merged dataset: {num_countries}")
    logger.info(f"Total years in merged dataset: {num_years}")

    mean_by_region = stats_summary["mean_by_region"]
    top_regions = mean_by_region.head(3)
    bottom_regions = mean_by_region.tail(3)
    logger.info("Top 3 regions by mean happiness score:")
    for region, value in top_regions.items():
        logger.info(f"  {region}: {value:.4f}")
    logger.info("Bottom 3 regions by mean happiness score:")
    for region, value in bottom_regions.items():
        logger.info(f"  {region}: {value:.4f}")

    interpretation = t_test_summary.get("interpretation", "No interpretation available.")
    logger.info(f"Pre/post-2020 t-test result: {interpretation}")

    correlations = correlation_summary["correlations"]
    corrected = [c for c in correlations if c["significant_bonferroni"]]
    if corrected:
        strongest = max(corrected, key=lambda r: abs(r["coef"]))
        logger.info(
            f"Most strongly correlated variable after Bonferroni correction: {strongest['variable']} "
            f"(coef = {strongest['coef']:.4f}, p = {strongest['p_value']:.6f})"
        )
    else:
        strongest = max(correlations, key=lambda r: abs(r["coef"])) if correlations else None
        if strongest is not None:
            logger.info(
                "No correlations remain significant after Bonferroni correction. "
                f"Strongest overall variable is {strongest['variable']} "
                f"(coef = {strongest['coef']:.4f}, p = {strongest['p_value']:.6f})."
            )
        else:
            logger.info("No correlation results were available to summarize.")


@flow
def happiness_pipeline() -> None:
    df = load_happiness_data()
    stats_summary = descriptive_statistics(df)
    create_plots(df)
    hypothesis_results = hypothesis_testing(df)
    correlation_summary = correlation_analysis(df)
    summary_report(
        df,
        stats_summary,
        hypothesis_results["pre_post_t_test"],
        correlation_summary,
    )


if __name__ == "__main__":
    happiness_pipeline()
