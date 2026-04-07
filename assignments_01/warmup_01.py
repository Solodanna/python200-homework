"""Warmup exercise solutions for Part 1."""

import pandas as pd
import numpy as np

if __name__ == "__main__":
    # --- Pandas ---

    # Pandas Q1
    data = {
        "name": ["Alice", "Bob", "Carol", "David", "Eve"],
        "grade": [85, 72, 90, 68, 95],
        "city": ["Boston", "Austin", "Boston", "Denver", "Austin"],
        "passed": [True, True, True, False, True],
    }
    df = pd.DataFrame(data)

    print("Pandas Q1: first three rows")
    print(df.head(3))
    print()

    print(f"Pandas Q1: shape = {df.shape}")
    print()

    print("Pandas Q1: dtypes")
    print(df.dtypes)
    print()

    # Pandas Q2
    passed_above_80 = df[(df["passed"]) & (df["grade"] > 80)]
    print("Pandas Q2: passed and grade above 80")
    print(passed_above_80)
    print()

    # Pandas Q3
    df["grade_curved"] = df["grade"] + 5
    print("Pandas Q3: updated DataFrame with grade_curved")
    print(df)
    print()

    # Pandas Q4
    df["name_upper"] = df["name"].str.upper()
    print("Pandas Q4: name and name_upper columns")
    print(df[["name", "name_upper"]])
    print()

    # Pandas Q5
    mean_grade_by_city = df.groupby("city", as_index=False)["grade"].mean()
    print("Pandas Q5: mean grade by city")
    print(mean_grade_by_city)
    print()

    # Pandas Q6
    df["city"] = df["city"].replace("Austin", "Houston")
    print("Pandas Q6: name and city columns after replacing Austin with Houston")
    print(df[["name", "city"]])
    print()

    # Pandas Q7
    top3_sorted = df.sort_values(by="grade", ascending=False).head(3)
    print("Pandas Q7: top 3 rows by grade descending")
    print(top3_sorted)
    print()

    # --- Matplotlib ---
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Matplotlib Q1
    x = [0, 1, 2, 3, 4, 5]
    y = [0, 1, 4, 9, 16, 25]
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.title("Squares")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Matplotlib Q2
    subjects = ["Math", "Science", "English", "History"]
    scores = [88, 92, 75, 83]
    plt.figure()
    plt.bar(subjects, scores, color='skyblue')
    plt.title("Subject Scores")
    plt.xlabel("Subject")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()

    # Matplotlib Q3
    x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
    x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
    plt.figure()
    plt.scatter(x1, y1, color='blue', label='Dataset 1')
    plt.scatter(x2, y2, color='red', label='Dataset 2')
    plt.title("Scatter Plot")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Matplotlib Q4
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x, y, marker='o')
    axes[0].set_title("Squares Line")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    axes[1].bar(subjects, scores, color='orange')
    axes[1].set_title("Subject Scores Bar")
    axes[1].set_xlabel("Subject")
    axes[1].set_ylabel("Score")

    plt.tight_layout()
    plt.show()

    # --- Descriptive Statistics ---

    # Descriptive Stats Q1
    data_stats = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
    print("Descriptive Stats Q1")
    print(f"Mean = {np.mean(data_stats)}")
    print(f"Median = {np.median(data_stats)}")
    print(f"Variance = {np.var(data_stats, ddof=0)}")
    print(f"Standard Deviation = {np.std(data_stats, ddof=0)}")
    print()

    # Descriptive Stats Q2
    random_scores = np.random.normal(65, 10, 500)
    plt.figure()
    plt.hist(random_scores, bins=20, color='green', edgecolor='black')
    plt.title("Distribution of Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Descriptive Stats Q3
    group_a = [55, 60, 63, 70, 68, 62, 58, 65]
    group_b = [75, 80, 78, 90, 85, 79, 82, 88]
    plt.figure()
    plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
    plt.title("Score Comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()

    # Descriptive Stats Q4
    normal_data = np.random.normal(50, 5, 200)
    skewed_data = np.random.exponential(10, 200)
    plt.figure()
    plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])
    plt.title("Distribution Comparison")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()
    # The exponential distribution is more skewed,
    # and the median is a more appropriate central tendency measure for it.
    # The normal distribution is symmetric, so the mean and median are both appropriate.

    # Descriptive Stats Q5
    data1 = [10, 12, 12, 16, 18]
    data2 = [10, 12, 12, 16, 150]
    import statistics
    print("Descriptive Stats Q5")
    print(f"data1 mean = {statistics.mean(data1)}")
    print(f"data1 median = {statistics.median(data1)}")
    print(f"data1 mode = {statistics.mode(data1)}")
    print(f"data2 mean = {statistics.mean(data2)}")
    print(f"data2 median = {statistics.median(data2)}")
    print(f"data2 mode = {statistics.mode(data2)}")
    # The median and mean differ for data2 because 150 is an outlier that pulls the mean upward.
    print()

    # --- Hypothesis Testing ---
    from scipy import stats

    # Hypothesis Q1
    group_a = [72, 68, 75, 70, 69, 73, 71, 74]
    group_b = [80, 85, 78, 83, 82, 86, 79, 84]
    t_stat, p_val = stats.ttest_ind(group_a, group_b)
    print("Hypothesis Q1")
    print(f"t-statistic = {t_stat}")
    print(f"p-value = {p_val}")
    print()

    # Hypothesis Q2
    alpha = 0.05
    if p_val < alpha:
        print("Hypothesis Q2: The difference is statistically significant at alpha = 0.05.")
    else:
        print("Hypothesis Q2: The difference is not statistically significant at alpha = 0.05.")
    print()

    # Hypothesis Q3
    before = [60, 65, 70, 58, 62, 67, 63, 66]
    after = [68, 70, 76, 65, 69, 72, 70, 71]
    t_stat_paired, p_val_paired = stats.ttest_rel(before, after)
    print("Hypothesis Q3")
    print(f"paired t-statistic = {t_stat_paired}")
    print(f"paired p-value = {p_val_paired}")
    print()

    # Hypothesis Q4
    scores = [72, 68, 75, 70, 69, 74, 71, 73]
    t_stat_one, p_val_one = stats.ttest_1samp(scores, 70)
    print("Hypothesis Q4")
    print(f"one-sample t-statistic = {t_stat_one}")
    print(f"one-sample p-value = {p_val_one}")
    print()

    # Hypothesis Q5
    t_stat_alt, p_val_alt = stats.ttest_ind(group_a, group_b, alternative='less')
    print("Hypothesis Q5")
    print(f"one-tailed p-value = {p_val_alt}")
    print()

    # Hypothesis Q6
    print("Hypothesis Q6: Conclusion")
    print("The students in group_b scored consistently higher than group_a, so the observed difference is unlikely to be due to chance at the 0.05 level.")
    print()

    # --- Correlation ---

    # Correlation Q1
    x_corr = [1, 2, 3, 4, 5]
    y_corr = [2, 4, 6, 8, 10]
    corr_matrix = np.corrcoef(x_corr, y_corr)
    print("Correlation Q1: correlation matrix")
    print(corr_matrix)
    print(f"Correlation Q1: coefficient = {corr_matrix[0, 1]}")
    # I expect the correlation to be 1.0 because y is a perfect linear function of x.
    print()

    # Correlation Q2
    from scipy.stats import pearsonr
    x2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y2 = [10, 9, 7, 8, 6, 5, 3, 4, 2, 1]
    coeff, pvalue = pearsonr(x2, y2)
    print("Correlation Q2")
    print(f"pearson coefficient = {coeff}")
    print(f"p-value = {pvalue}")
    print()

    # Correlation Q3
    people = {
        "height": [160, 165, 170, 175, 180],
        "weight": [55, 60, 65, 72, 80],
        "age": [25, 30, 22, 35, 28],
    }
    df_people = pd.DataFrame(people)
    print("Correlation Q3")
    print(df_people.corr())
    print()

    # Correlation Q4
    x_neg = [10, 20, 30, 40, 50]
    y_neg = [90, 75, 60, 45, 30]
    plt.figure()
    plt.scatter(x_neg, y_neg, color='purple')
    plt.title("Negative Correlation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

    # Correlation Q5
    plt.figure()
    sns.heatmap(df_people.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
    print()

    # --- NumPy ---

    # NumPy Q1
    arr1d = np.array([10, 20, 30, 40, 50])
    print("NumPy Q1: array")
    print(arr1d)
    print(f"NumPy Q1: shape = {arr1d.shape}")
    print(f"NumPy Q1: dtype = {arr1d.dtype}")
    print(f"NumPy Q1: ndim = {arr1d.ndim}")
    print()

    # NumPy Q2
    arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("NumPy Q2: array")
    print(arr2d)
    print(f"NumPy Q2: shape = {arr2d.shape}")
    print(f"NumPy Q2: size = {arr2d.size}")
    print()

    # NumPy Q3
    top_left_2x2 = arr2d[:2, :2]
    print("NumPy Q3: top-left 2x2 block")
    print(top_left_2x2)
    print()

    # NumPy Q4
    zeros_3x4 = np.zeros((3, 4))
    ones_2x5 = np.ones((2, 5))
    print("NumPy Q4: 3x4 zeros array")
    print(zeros_3x4)
    print()

    print("NumPy Q4: 2x5 ones array")
    print(ones_2x5)
    print()

    # NumPy Q5
    arange_arr = np.arange(0, 50, 5)
    print("NumPy Q5: array")
    print(arange_arr)
    print(f"NumPy Q5: shape = {arange_arr.shape}")
    print(f"NumPy Q5: mean = {arange_arr.mean()}")
    print(f"NumPy Q5: sum = {arange_arr.sum()}")
    print(f"NumPy Q5: std = {arange_arr.std()}")
    print()

    # NumPy Q6
    random_norm = np.random.normal(loc=0, scale=1, size=200)
    print("NumPy Q6: random normal mean")
    print(random_norm.mean())
    print("NumPy Q6: random normal std")
    print(random_norm.std())
    print()

    # --- Pipelines ---

    def create_series(arr):
        return pd.Series(arr, name="values")

    def clean_data(series):
        return series.dropna()

    def summarize_data(series):
        return {
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "mode": series.mode()[0],
        }

    def data_pipeline(arr):
        series = create_series(arr)
        cleaned = clean_data(series)
        return summarize_data(cleaned)

    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    summary = data_pipeline(arr)
    print("Pipelines Q1")
    for key, value in summary.items():
        print(f"{key} = {value}")
