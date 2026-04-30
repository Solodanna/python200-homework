import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	ConfusionMatrixDisplay,
	accuracy_score,
	classification_report,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# --- Setup ---
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Task 1: Load and Explore ---
# Column names from the UCI Spambase dataset documentation.
feature_names = [
	"word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our",
	"word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail",
	"word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses",
	"word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
	"word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp",
	"word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs",
	"word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85",
	"word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
	"word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re",
	"word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(",
	"char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#", "capital_run_length_average",
	"capital_run_length_longest", "capital_run_length_total", "spam_label",
]

spambase_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
df = pd.read_csv(spambase_url, header=None, names=feature_names)

print("=== Task 1: Load and Explore ===")
print(f"Dataset shape: {df.shape}")
print(f"Number of emails: {len(df)}")

class_counts = df["spam_label"].value_counts().sort_index()
class_props = df["spam_label"].value_counts(normalize=True).sort_index()
print("Class counts (0=ham, 1=spam):")
print(class_counts)
print("Class proportions (0=ham, 1=spam):")
print(class_props)

majority_class_accuracy = max(class_props)
print(f"Majority-class baseline accuracy: {majority_class_accuracy:.4f}")
# Because classes are not perfectly balanced, a high raw accuracy can be misleading;
# compare against the baseline and also inspect precision/recall for each class.

# Boxplots for selected features split by class.
for feature in ["word_freq_free", "char_freq_!", "capital_run_length_total"]:
	fig, ax = plt.subplots(figsize=(8, 5))
	df.boxplot(column=feature, by="spam_label", ax=ax)
	ax.set_title(f"{feature} by Spam Label")
	ax.set_xlabel("spam_label (0=ham, 1=spam)")
	ax.set_ylabel(feature)
	fig.suptitle("")
	fig.tight_layout()
	fig.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{feature}.png"), dpi=150)
	plt.close(fig)

print("Saved boxplots for word_freq_free, char_freq_!, and capital_run_length_total.")

zero_rate_word_features = (df[[c for c in df.columns if c.startswith("word_freq_")]] == 0).mean().mean()
print(f"Average zero-rate across word_freq_* features: {zero_rate_word_features:.4f}")
print("Feature scale snapshot (min/max for selected columns):")
print(df[["word_freq_free", "char_freq_!", "capital_run_length_total"]].agg(["min", "max"]))
# Heavy skew toward zero means the feature space is sparse: many words simply do not appear in most emails.
# Scale differences are large because frequency features are percentages while capital-run statistics can be huge counts.
# This matters for distance- and magnitude-based models (KNN, logistic regression), which can be dominated by large-scale features.


# --- Task 2: Prepare Your Data ---
print("\n=== Task 2: Prepare Your Data ===")
X = df.drop(columns=["spam_label"])
y = df["spam_label"]

# Stratified split preserves class balance between train and test.
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Fit scaler on train only to avoid test leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit PCA on scaled train data only.
pca = PCA()
pca.fit(X_train_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n = int(np.argmax(cumulative_variance >= 0.90) + 1)
print(f"Number of PCA components to reach 90% explained variance: n={n}")

fig_var, ax_var = plt.subplots(figsize=(8, 5))
ax_var.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance)
ax_var.axhline(0.90, linestyle="--", color="red", label="90% variance")
ax_var.axvline(n, linestyle="--", color="green", label=f"n={n}")
ax_var.set_title("Spambase PCA Cumulative Explained Variance")
ax_var.set_xlabel("Number of components")
ax_var.set_ylabel("Cumulative explained variance ratio")
ax_var.legend()
fig_var.tight_layout()
fig_var.savefig(os.path.join(OUTPUT_DIR, "pca_cumulative_variance.png"), dpi=150)
plt.close(fig_var)

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]


# --- Task 3: A Classifier Comparison ---
print("\n=== Task 3: Classifier Comparison ===")


def evaluate_classifier(name, model, Xtr, Xte, ytr, yte):
	"""Fit model, print metrics, and return accuracy/predictions for tracking best model."""
	model.fit(Xtr, ytr)
	preds = model.predict(Xte)
	acc = accuracy_score(yte, preds)
	print(f"\n{name}")
	print(f"Accuracy: {acc:.4f}")
	print("Classification report:")
	print(classification_report(yte, preds, digits=4))
	return acc, preds, model


results = {}

# KNN on unscaled data
acc_knn_unscaled, pred_knn_unscaled, model_knn_unscaled = evaluate_classifier(
	"KNN (k=5) - Unscaled",
	KNeighborsClassifier(n_neighbors=5),
	X_train,
	X_test,
	y_train,
	y_test,
)
results["KNN unscaled"] = {
	"accuracy": acc_knn_unscaled,
	"preds": pred_knn_unscaled,
	"model": model_knn_unscaled,
	"X_test": X_test,
}

# KNN on scaled and PCA data
acc_knn_scaled, pred_knn_scaled, model_knn_scaled = evaluate_classifier(
	"KNN (k=5) - Scaled",
	KNeighborsClassifier(n_neighbors=5),
	X_train_scaled,
	X_test_scaled,
	y_train,
	y_test,
)
results["KNN scaled"] = {
	"accuracy": acc_knn_scaled,
	"preds": pred_knn_scaled,
	"model": model_knn_scaled,
	"X_test": X_test_scaled,
}

acc_knn_pca, pred_knn_pca, model_knn_pca = evaluate_classifier(
	f"KNN (k=5) - PCA first {n} components",
	KNeighborsClassifier(n_neighbors=5),
	X_train_pca,
	X_test_pca,
	y_train,
	y_test,
)
results["KNN PCA"] = {
	"accuracy": acc_knn_pca,
	"preds": pred_knn_pca,
	"model": model_knn_pca,
	"X_test": X_test_pca,
}

print(
	f"\nKNN comparison: scaled ({acc_knn_scaled:.4f}) vs PCA ({acc_knn_pca:.4f})"
)

# Decision tree depth experiment
print("\nDecision Tree depth sweep (train vs test accuracy):")
depth_results = []
for depth in [3, 5, 10, None]:
	tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
	tree.fit(X_train, y_train)
	train_acc = accuracy_score(y_train, tree.predict(X_train))
	test_acc = accuracy_score(y_test, tree.predict(X_test))
	depth_results.append((depth, train_acc, test_acc))
	print(f"max_depth={depth}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

# Pick depth=5 for a strong balance: good test performance without extreme train/test gap.
chosen_depth = 5
acc_tree, pred_tree, model_tree = evaluate_classifier(
	f"DecisionTree (max_depth={chosen_depth})",
	DecisionTreeClassifier(max_depth=chosen_depth, random_state=42),
	X_train,
	X_test,
	y_train,
	y_test,
)
results[f"DecisionTree depth {chosen_depth}"] = {
	"accuracy": acc_tree,
	"preds": pred_tree,
	"model": model_tree,
	"X_test": X_test,
}

# Random forest
acc_rf, pred_rf, model_rf = evaluate_classifier(
	"RandomForest (n_estimators=300)",
	RandomForestClassifier(n_estimators=300, random_state=42),
	X_train,
	X_test,
	y_train,
	y_test,
)
results["RandomForest"] = {
	"accuracy": acc_rf,
	"preds": pred_rf,
	"model": model_rf,
	"X_test": X_test,
}

# Logistic regression on scaled and PCA data
acc_lr_scaled, pred_lr_scaled, model_lr_scaled = evaluate_classifier(
	"LogisticRegression (C=1.0) - Scaled",
	LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
	X_train_scaled,
	X_test_scaled,
	y_train,
	y_test,
)
results["LogReg scaled"] = {
	"accuracy": acc_lr_scaled,
	"preds": pred_lr_scaled,
	"model": model_lr_scaled,
	"X_test": X_test_scaled,
}

acc_lr_pca, pred_lr_pca, model_lr_pca = evaluate_classifier(
	f"LogisticRegression (C=1.0) - PCA first {n} components",
	LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
	X_train_pca,
	X_test_pca,
	y_train,
	y_test,
)
results["LogReg PCA"] = {
	"accuracy": acc_lr_pca,
	"preds": pred_lr_pca,
	"model": model_lr_pca,
	"X_test": X_test_pca,
}

print(
	f"\nLogistic regression comparison: scaled ({acc_lr_scaled:.4f}) vs PCA ({acc_lr_pca:.4f})"
)

# Summary comment for interpretation:
# Compare all accuracies and reports together. For KNN/logistic regression, PCA may help or hurt depending on
# whether variance-compressing loses class-discriminative detail. For spam filtering, accuracy alone is insufficient:
# false positives (ham flagged as spam) can be costly for users, while false negatives allow spam through.
# In many real systems, minimizing false positives is prioritized while keeping false negatives manageable.
print("\nTask 3 summary:")
print("- RandomForest has the highest accuracy among tested models on this split.")
print("- For KNN and LogisticRegression here, scaled non-PCA features slightly outperform PCA-reduced features.")
print("- This matches the idea that PCA can remove some predictive detail even when it keeps most variance.")
print("- For spam filtering, I prioritize minimizing false positives so legitimate emails are not incorrectly blocked.")

# Best model tracking and confusion matrix
best_model_name = max(results, key=lambda key: results[key]["accuracy"])
best_info = results[best_model_name]
print(f"\nBest model by test accuracy: {best_model_name} ({best_info['accuracy']:.4f})")

fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
	y_test,
	best_info["preds"],
	display_labels=["ham (0)", "spam (1)"],
	cmap="Blues",
	ax=ax_cm,
	colorbar=False,
)
ax_cm.set_title(f"Best Model Confusion Matrix: {best_model_name}")
fig_cm.tight_layout()
fig_cm.savefig(os.path.join(OUTPUT_DIR, "best_model_confusion_matrix.png"), dpi=150)
plt.close(fig_cm)

print("Saved best model confusion matrix to outputs/best_model_confusion_matrix.png")

# Error balance for best model
tn, fp, fn, tp = pd.crosstab(
	y_test,
	best_info["preds"],
	rownames=["actual"],
	colnames=["pred"],
	dropna=False,
).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy().ravel()
print(f"Best model errors -> false positives: {fp}, false negatives: {fn}")
if fp > fn:
	print("This model makes more false positives than false negatives.")
elif fn > fp:
	print("This model makes more false negatives than false positives.")
else:
	print("This model makes false positives and false negatives at the same rate.")


# --- Task 4: Cross-Validation ---
print("\n=== Task 4: Cross-Validation (cv=5 on Training Data) ===")

# Use pipelines for scaled/PCA models so scaling/PCA are fit inside each CV fold.
cv_models = {
	"KNN unscaled": KNeighborsClassifier(n_neighbors=5),
	"KNN scaled": Pipeline([
		("scaler", StandardScaler()),
		("classifier", KNeighborsClassifier(n_neighbors=5)),
	]),
	"KNN PCA": Pipeline([
		("scaler", StandardScaler()),
		("pca", PCA(n_components=n)),
		("classifier", KNeighborsClassifier(n_neighbors=5)),
	]),
	f"DecisionTree depth {chosen_depth}": DecisionTreeClassifier(max_depth=chosen_depth, random_state=42),
	"RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
	"LogReg scaled": Pipeline([
		("scaler", StandardScaler()),
		("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")),
	]),
	"LogReg PCA": Pipeline([
		("scaler", StandardScaler()),
		("pca", PCA(n_components=n)),
		("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")),
	]),
}

cv_summary = {}
for model_name, model in cv_models.items():
	fold_scores = cross_val_score(model, X_train, y_train, cv=5)
	mean_score = float(fold_scores.mean())
	std_score = float(fold_scores.std())
	cv_summary[model_name] = {"mean": mean_score, "std": std_score}
	print(f"{model_name:24s} -> mean={mean_score:.4f}, std={std_score:.4f}")

most_accurate_cv = max(cv_summary, key=lambda k: cv_summary[k]["mean"])
most_stable_cv = min(cv_summary, key=lambda k: cv_summary[k]["std"])
print(f"Most accurate by CV mean: {most_accurate_cv} ({cv_summary[most_accurate_cv]['mean']:.4f})")
print(f"Most stable by CV std: {most_stable_cv} ({cv_summary[most_stable_cv]['std']:.4f})")

split_ranking = sorted(results.items(), key=lambda kv: kv[1]["accuracy"], reverse=True)
cv_ranking = sorted(cv_summary.items(), key=lambda kv: kv[1]["mean"], reverse=True)
print("Single-split ranking (top to bottom):")
print(" -> ".join([name for name, _ in split_ranking]))
print("CV ranking (top to bottom):")
print(" -> ".join([name for name, _ in cv_ranking]))
print("Ranking comparison note: if the top models are similar across both lists, the single-split conclusion is likely robust.")


# --- Task 5: Building a Prediction Pipeline ---
print("\n=== Task 5: Prediction Pipelines ===")

# Best tree-based model from Task 3: RandomForest.
tree_pipeline = Pipeline([
	("classifier", RandomForestClassifier(n_estimators=300, random_state=42)),
])
tree_pipeline.fit(X_train, y_train)
tree_pipeline_preds = tree_pipeline.predict(X_test)
tree_pipeline_acc = accuracy_score(y_test, tree_pipeline_preds)
print("\nTree pipeline (RandomForest) classification report:")
print(classification_report(y_test, tree_pipeline_preds, digits=4))
print(f"Tree pipeline accuracy: {tree_pipeline_acc:.4f}")
print(f"Manual RandomForest accuracy from Task 3: {results['RandomForest']['accuracy']:.4f}")

# Best non-tree-based model from Task 3: LogisticRegression on scaled features.
# PCA is not included because Task 3 showed scaled non-PCA outperformed PCA for logistic regression.
non_tree_pipeline = Pipeline([
	("scaler", StandardScaler()),
	("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")),
])
non_tree_pipeline.fit(X_train, y_train)
non_tree_pipeline_preds = non_tree_pipeline.predict(X_test)
non_tree_pipeline_acc = accuracy_score(y_test, non_tree_pipeline_preds)
print("\nNon-tree pipeline (Scaled LogisticRegression) classification report:")
print(classification_report(y_test, non_tree_pipeline_preds, digits=4))
print(f"Non-tree pipeline accuracy: {non_tree_pipeline_acc:.4f}")
print(f"Manual LogReg scaled accuracy from Task 3: {results['LogReg scaled']['accuracy']:.4f}")

# Pipeline structure comment:
# These pipelines do not have identical structure because tree models do not need scaling/PCA,
# while logistic regression benefits from consistent feature scaling.
# Packaging as pipelines improves reproducibility, prevents preprocessing mistakes/leakage,
# and makes handoff/deployment easier because one object encapsulates preprocessing + prediction.
