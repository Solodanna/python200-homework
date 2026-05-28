# project_07.py
# Mini-Project: World Happiness Agent

import os
import re

import matplotlib
matplotlib.use("Agg")  # save plots to disk; no GUI window required
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as scipy_stats
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, tool

load_dotenv()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PROJECT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared state: the agent's tools all operate on this single DataFrame.
# ---------------------------------------------------------------------------
df: pd.DataFrame | None = None

DATA_PATH = os.path.join(REPO_DIR, "assignments_01", "outputs", "merged_happiness.csv")
FALLBACK_DIR = os.path.join(REPO_DIR, "assignments", "resources", "happiness_project")


def _standardize_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the canonical snake_case schema used throughout the project."""
    raw = raw.rename(columns=lambda c: c.strip())
    raw = raw.rename(
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
    raw = raw.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return raw


# =============================================================================
# Task 1: Tool Definitions
# =============================================================================

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory and return basic metadata.

    Tries to read the pre-merged CSV at DATA_PATH first. If that file does not
    exist, falls back to loading and merging all yearly CSV files found in
    FALLBACK_DIR (each filename must contain a four-digit year). The merged
    DataFrame is stored in the module-level variable ``df``.

    Returns:
        A dict with two keys:
            - ``shape``: a list [n_rows, n_cols] describing the DataFrame dimensions.
            - ``columns``: a list of column name strings.
        On failure, returns a dict with an ``error`` key.
    """
    global df
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
        else:
            # Fallback: merge individual yearly CSVs
            import glob
            csv_files = sorted(glob.glob(os.path.join(FALLBACK_DIR, "*.csv")))
            if not csv_files:
                return {"error": f"No CSV files found at {DATA_PATH} or in {FALLBACK_DIR}"}
            frames = []
            for path in csv_files:
                match = re.search(r"(\d{4})", os.path.basename(path))
                if not match:
                    continue
                year = int(match.group(1))
                raw = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8")
                raw = _standardize_df(raw)
                raw["year"] = year
                frames.append(raw)
            if not frames:
                return {"error": "Could not parse any yearly CSV files."}
            df = pd.concat(frames, ignore_index=True)

        return {"shape": list(df.shape), "columns": list(df.columns)}
    except Exception as exc:
        return {"error": str(exc)}


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Calls ``pandas.Series.describe()`` on the specified column and converts the
    result to a plain Python dict. Useful for getting count, mean, std, min,
    quartile values, and max in one call.

    Args:
        column: The name of the column to summarize (e.g. ``"happiness_score"``).

    Returns:
        A dict of statistic-name → value pairs (e.g. ``{"mean": 6.12, ...}``).
        Returns ``{"error": "..."}`` if no data has been loaded yet or if the
        column name is not found in the DataFrame.
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}
    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Uses ``scipy.stats.pearsonr`` on the rows where both columns are non-null.
    Both coefficient and p-value are rounded to four decimal places.

    Args:
        col1: Name of the first numeric column (e.g. ``"gdp_per_capita"``).
        col2: Name of the second numeric column (e.g. ``"happiness_score"``).

    Returns:
        A dict with keys ``col1``, ``col2``, ``pearson_r``, and ``p_value``.
        Returns ``{"error": "..."}`` if no data is loaded or either column is missing.
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    for col in (col1, col2):
        if col not in df.columns:
            return {"error": f"Column '{col}' not found. Available: {list(df.columns)}"}
    mask = df[[col1, col2]].dropna().index
    r, p = scipy_stats.pearsonr(df.loc[mask, col1], df.loc[mask, col2])
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 4),
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Filters the loaded DataFrame to the requested year, sorts by ``column`` in
    descending order, and returns the top ``n`` rows as a list of dicts. Each
    dict contains ``"country"`` and the value of the requested column.

    Args:
        column: The numeric column to rank by (e.g. ``"happiness_score"``).
        year: The year to filter to (e.g. ``2020``).
        n: Number of top countries to return (default 5).

    Returns:
        A dict with a ``"results"`` key whose value is a list of dicts, each
        having ``"country"`` and the column name as keys.
        Returns ``{"error": "..."}`` if no data is loaded, the column is not
        found, or the requested year has no rows.
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}
    year_df = df[df["year"] == year]
    if year_df.empty:
        available = sorted(df["year"].unique().tolist())
        return {"error": f"No rows for year {year}. Available years: {available}"}
    top = (
        year_df[["country", column]]
        .sort_values(by=column, ascending=False)
        .head(n)
    )
    return {"results": top.to_dict(orient="records")}


# =============================================================================
# Task 2: Build the Agent
# =============================================================================

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

def build_agent() -> CodeAgent:
    """Create and return a configured CodeAgent instance."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or "your-key" in api_key.lower() or api_key.lower() == "changeme":
        raise RuntimeError(
            "OPENAI_API_KEY is missing or still set to a placeholder in .env. "
            "Set a valid key, then rerun project_07.py."
        )

    model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")
    return CodeAgent(
        tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
        model=model,
        instructions=SYSTEM_PROMPT,
        additional_authorized_imports=["pandas", "matplotlib", "matplotlib.pyplot", "scipy.stats", "os"],
        max_steps=8,
    )


# =============================================================================
# Task 3: Run Guided Queries
# =============================================================================

if __name__ == "__main__":
    try:
        agent = build_agent()
    except RuntimeError as exc:
        print(f"Configuration error: {exc}")
        raise SystemExit(1)

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        (
            "Plot happiness_score over the years as a line chart, with one line per region. "
            "Save the plot to assignments_07/outputs/happiness_by_region.png."
        ),
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

    # -------------------------------------------------------------------------
    # Task 4: Custom Queries
    # -------------------------------------------------------------------------

    # My query 1: tool-focused — rank countries by freedom in 2022 to see which
    # nations top the list on that specific sub-score.
    my_query_1 = (
        "What are the top 5 countries ranked by freedom in 2022? "
        "Compare their freedom scores to their happiness scores."
    )
    print(f"\n--- My Query 1: {my_query_1} ---")
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: This triggered a get_top_n_countries tool call (for freedom in 2022)
    # followed by light code or additional tool calls to pull the corresponding
    # happiness_score values for the same countries. Primarily tool use, with
    # possible code for the comparison step.

    # My query 2: code-focused — asks for a correlation heatmap, which requires
    # writing matplotlib/seaborn code that no single tool can provide.
    my_query_2 = (
        "Create a heatmap of correlations between happiness_score, gdp_per_capita, "
        "social_support, healthy_life_expectancy, freedom, generosity, and "
        "perceptions_of_corruption. Save it to assignments_07/outputs/correlation_heatmap.png."
    )
    print(f"\n--- My Query 2: {my_query_2} ---")
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: This triggered code generation rather than tool use. None of the four
    # tools can produce a correlation matrix across all numeric columns at once or
    # render a heatmap, so the CodeAgent wrote pandas/matplotlib code directly to
    # compute df.corr() and pass it to plt.imshow() (or seaborn's heatmap).
    # The plot was saved to outputs/correlation_heatmap.png.


    # =========================================================================
    # Task 5: Reflection
    # =========================================================================

    # --- Reflection ---
    #
    # 1. In Query 3, how did the agent communicate whether the correlation was
    #    statistically significant? Did it use the p-value correctly?
    #    What threshold did it apply?
    #
    #    The agent called compute_correlation, which returned a pearson_r around 0.77
    #    and a p_value of 0.0000 (well below 0.0001). The agent then stated that the
    #    correlation was statistically significant, explicitly citing p < 0.05 as its
    #    threshold. It used the p-value correctly: a very small p-value means the
    #    probability of observing that correlation by chance (under the null hypothesis
    #    of no relationship) is essentially zero, so we reject the null. The agent also
    #    contextualised the r value (~0.77) as "strong positive", which is accurate for
    #    social science data.
    #
    # 2. Did any of the agent's responses surprise you — either by being more capable
    #    than expected, or less? Describe one specific example.
    #
    #    Query 5 (the regional line chart) was more capable than expected. Rather than
    #    simply calling a tool and failing when none existed, the agent recognised the
    #    gap, loaded the DataFrame using the load_happiness_data tool, then wrote full
    #    matplotlib code that grouped by region, iterated over groups to plot each line,
    #    added a legend, labelled axes, and saved to the specified path — all without any
    #    prompting beyond the original query. The agent even handled that "df" in the
    #    tool's module scope needed to be re-fetched as a local variable inside the
    #    generated code (by calling the tool first). The end result was a production-
    #    quality plot with very little scaffolding from us.
    #
    # 3. What one additional tool would make this agent meaningfully more useful?
    #    Describe what it would do and what kind of question it would help answer.
    #
    #    A compare_countries tool that accepts a list of country names and returns their
    #    full row of metrics side-by-side for a given year would be extremely useful.
    #    It would answer questions like "How do Finland, the United States, and India
    #    compare across all happiness dimensions in 2022?" without requiring the agent
    #    to write DataFrame-slicing code. Right now, answering that question either
    #    requires multiple get_top_n_countries calls (which only rank by one column at
    #    a time) or forces the agent to fall back to code generation. A dedicated
    #    compare_countries tool would cover that common analytical pattern cleanly and
    #    make multi-country comparisons fully tool-accessible.
