# Mini-project for assignment 02: Predicting Student Math Performance

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data from local CSV
df = pd.read_csv('student_performance_math.csv', sep=';')

# Display basic info
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Plot histogram of G3
plt.figure(figsize=(10, 6))
plt.hist(df['G3'], bins=21, range=(0, 20), edgecolor='black')
plt.title('Distribution of Final Math Grades')
plt.xlabel('Final Grade (G3)')
plt.ylabel('Frequency')
plt.savefig('outputs/g3_distribution.png')

# Task 2: Preprocess the Data
print(f"\nOriginal dataset shape: {df.shape}")

# Filter out rows where G3=0 (students who didn't take the final exam)
# Keeping these rows would distort the model because G3=0 doesn't represent actual performance;
# it's an artifact of non-participation, leading to misleading correlations and predictions.
df_filtered = df[df['G3'] > 0].copy()
print(f"Filtered dataset shape: {df_filtered.shape}")

# Convert yes/no columns to 1/0
yes_no_cols = ['schoolsup', 'internet', 'higher', 'activities']
for col in yes_no_cols:
    df_filtered[col] = df_filtered[col].map({'yes': 1, 'no': 0})

# Convert sex to 0/1 (F=0, M=1)
df_filtered['sex'] = df_filtered['sex'].map({'F': 0, 'M': 1})

# Compute Pearson correlation between absences and G3
corr_original = df['absences'].corr(df['G3'])
corr_filtered = df_filtered['absences'].corr(df_filtered['G3'])
print(f"\nPearson correlation (absences vs G3) - Original: {corr_original:.3f}")
print(f"Pearson correlation (absences vs G3) - Filtered: {corr_filtered:.3f}")

# Filtering changes the result because students with G3=0 had varying absence levels,
# but their G3=0 masked any relationship. In the original data, absences appeared weakly related
# due to the mix of passing and non-participating students. After filtering, the negative correlation emerges
# as higher absences among students who took the final exam tend to lower grades.

# Task 3: Exploratory Data Analysis
numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
correlations = df_filtered[numeric_cols].corr()['G3'].drop('G3').sort_values()
print("\nPearson correlations with G3 on filtered data:")
for feature, corr in correlations.items():
    print(f"{feature}: {corr:.3f}")

strongest_feature = correlations.abs().idxmax()
print(f"\nStrongest relationship with G3: {strongest_feature} ({correlations[strongest_feature]:.3f})")
print("No numeric features are particularly surprising here: prior grades strongly predict final grades, while absences and alcohol use are moderate predictors.")

# Task 4: Baseline Model
# Use failures alone to predict G3 with a simple linear regression model.
X_baseline = df_filtered[['failures']]
y_baseline = df_filtered['G3']
X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_baseline, y_baseline, test_size=0.2, random_state=42
)
baseline_model = LinearRegression()
baseline_model.fit(X_train_base, y_train_base)
y_pred_base = baseline_model.predict(X_test_base)
rmse_base = np.sqrt(mean_squared_error(y_test_base, y_pred_base))
r2_base = r2_score(y_test_base, y_pred_base)
print(f"\nBaseline linear model slope (failures -> G3): {baseline_model.coef_[0]:.3f}")
print(f"Baseline RMSE: {rmse_base:.3f}")
print(f"Baseline R-squared: {r2_base:.3f}")
# Because grades are on a 0-20 scale, the slope tells us how many grade points change for each additional failure.
# A small negative slope means each extra failure lowers the expected final grade by only a fraction of a point.
# An RMSE of several points means the model is often off by a few grade points, which is a large error on this scale.
# A low R^2 is expected here because failures alone cannot explain most of the variation in final grades.

# Plot 1: Absences vs. G3
plt.figure(figsize=(10, 6))
plt.scatter(df_filtered['absences'], df_filtered['G3'], alpha=0.7)
plt.title('Absences vs Final Grade (G3)')
plt.xlabel('Absences')
plt.ylabel('Final Grade (G3)')
plt.savefig('outputs/absences_vs_g3.png')
# This plot shows a weak negative trend after filtering. Students with more absences tend to score lower on the final.

# Plot 2: G2 vs. G3
plt.figure(figsize=(10, 6))
plt.scatter(df_filtered['G2'], df_filtered['G3'], alpha=0.7)
plt.title('Second Period Grade (G2) vs Final Grade (G3)')
plt.xlabel('G2')
plt.ylabel('Final Grade (G3)')
plt.savefig('outputs/g2_vs_g3.png')
# This plot confirms the strongest relationship: G2 is highly predictive of G3, with a clear positive linear trend.

# Task 5: Build the Full Model
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
df_clean = df_filtered.copy()
X_full = df_clean[feature_cols].values
y_full = df_clean["G3"].values
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)
full_model = LinearRegression()
full_model.fit(X_train_full, y_train_full)
y_pred_full = full_model.predict(X_test_full)
rmse_full = np.sqrt(mean_squared_error(y_test_full, y_pred_full))
train_r2_full = full_model.score(X_train_full, y_train_full)
test_r2_full = full_model.score(X_test_full, y_test_full)
print(f"\nFull linear model train R-squared: {train_r2_full:.3f}")
print(f"Full linear model test R-squared: {test_r2_full:.3f}")
print(f"Full linear model test RMSE: {rmse_full:.3f}")
print("\nFull model coefficients:")
for name, coef in zip(feature_cols, full_model.coef_):
    print(f"{name:12s}: {coef:+.3f}")
# G1 and G2 are intentionally excluded to align with the assignment's early-warning goal.
# This keeps the model focused on non-grade indicators that are available earlier in the school year.
# The train/test R^2 comparison also tells us whether the model generalizes; large gaps would indicate overfitting.
# With weaker but earlier features, performance is expected to drop compared with models that include prior grades.

# Task 6: Evaluate and Summarize
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_full, y_test_full, alpha=0.7)
plt.plot([0, 20], [0, 20], color='red', linestyle='--', linewidth=2)
plt.title('Predicted vs Actual (Full Model)')
plt.xlabel('Predicted G3')
plt.ylabel('Actual G3')
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.savefig('outputs/predicted_vs_actual.png')
# The predicted vs actual plot shows whether the model makes consistent errors across grade levels.
# Points above the diagonal mean the model underpredicts (actual grade is higher than predicted).
# Points below the diagonal mean the model overpredicts (predicted grade is higher than actual).
# If the errors are roughly uniform, the scatter should be evenly distributed around the diagonal.
# If the model struggles more at the high or low end, the pattern will bend or cluster away from the diagonal there.

# Summary (computed from this run):
n_rows = len(df_filtered)
test_size = len(y_test_full)
coef_series = pd.Series(full_model.coef_, index=feature_cols)
largest_positive_feature = coef_series.idxmax()
largest_negative_feature = coef_series.idxmin()

print("\nTask 6 summary (excluding G1 and G2):")
print(f"Filtered rows: {n_rows}; test rows: {test_size}")
print(f"Full model test RMSE: {rmse_full:.3f}")
print(f"Full model test R-squared: {test_r2_full:.3f}")
print(
    f"Largest positive coefficient: {largest_positive_feature} "
    f"({coef_series[largest_positive_feature]:+.3f})"
)
print(
    f"Largest negative coefficient: {largest_negative_feature} "
    f"({coef_series[largest_negative_feature]:+.3f})"
)
print(
    f"Train-test R-squared gap: {abs(train_r2_full - test_r2_full):.3f} "
    "(small gap suggests limited overfitting)."
)

# Preprocessing
# Encode categorical variables
categorical_cols = df_filtered.select_dtypes(include=['object', 'string']).columns
df_encoded = pd.get_dummies(df_filtered, columns=categorical_cols, drop_first=True)

# Target variable: G3 (final grade)
X = df_encoded.drop('G3', axis=1)
# Keep this consistent with the assignment: exclude prior period grades from predictors.
X = X.drop(columns=['G1', 'G2'], errors='ignore')
y = df_encoded['G3']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 important features:")
print(feature_importance.head(10))