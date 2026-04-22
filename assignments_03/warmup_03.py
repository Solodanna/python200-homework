import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Ensure output folder exists when this file is run directly.
import os
os.makedirs('outputs', exist_ok=True)


# --- Preprocessing ---
# Q1
print("\n=== Preprocessing Q1 ===")
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Q2
print("\n=== Preprocessing Q2 ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Column means of X_train_scaled (should be near 0):")
print(np.mean(X_train_scaled, axis=0))
# Fit on X_train only to avoid leaking information from the test set into preprocessing.


# --- KNN ---
# Q1
print("\n=== KNN Q1 (Unscaled Data, k=5) ===")
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_knn_unscaled = knn_unscaled.predict(X_test)
acc_knn_unscaled = accuracy_score(y_test, y_pred_knn_unscaled)
print(f"Accuracy (unscaled): {acc_knn_unscaled:.4f}")
print("Classification report (unscaled):")
print(classification_report(y_test, y_pred_knn_unscaled, target_names=iris.target_names))

# Q2
print("\n=== KNN Q2 (Scaled Data, k=5) ===")
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)
acc_knn_scaled = accuracy_score(y_test, y_pred_knn_scaled)
print(f"Accuracy (scaled): {acc_knn_scaled:.4f}")
# In this run, scaling slightly hurt KNN accuracy (0.9333 vs 1.0000), likely because Iris features are already fairly well-scaled and this split is small.

# Q3
print("\n=== KNN Q3 (5-fold CV on Unscaled Training Data, k=5) ===")
cv_scores_k5 = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_train, y_train, cv=5)
print("Fold scores:", cv_scores_k5)
print(f"Mean CV score: {cv_scores_k5.mean():.4f}")
print(f"Std CV score: {cv_scores_k5.std():.4f}")
# This is more trustworthy than one split because it averages across folds and is less sensitive to one lucky/unlucky split.

# Q4
print("\n=== KNN Q4 (Choose k with 5-fold CV on Unscaled Training Data) ===")
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
cv_means = []
for k in k_values:
	scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5)
	mean_score = scores.mean()
	cv_means.append(mean_score)
	print(f"k={k:2d} -> mean CV accuracy={mean_score:.4f}")

best_idx = int(np.argmax(cv_means))
best_k = k_values[best_idx]
best_k_score = cv_means[best_idx]
print(f"Chosen k: {best_k} (highest mean CV accuracy: {best_k_score:.4f})")
# Choose the k with the highest mean CV score; if tied, prefer the larger k for a slightly smoother, less noisy boundary.


# --- Classifier Evaluation ---
# Q1
print("\n=== Classifier Evaluation Q1 (KNN Confusion Matrix) ===")
cm_knn = confusion_matrix(y_test, y_pred_knn_unscaled)
print("Confusion matrix:\n", cm_knn)

fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=iris.target_names)
disp.plot(cmap='Blues', ax=ax_cm, colorbar=False)
ax_cm.set_title('KNN (k=5) Confusion Matrix - Iris')
fig_cm.tight_layout()
fig_cm.savefig('outputs/knn_confusion_matrix.png', dpi=150)
plt.close(fig_cm)
# On this split there is no confusion: all species are classified correctly.


# --- The sklearn API: Decision Trees ---
# Q1
print("\n=== Decision Trees Q1 (max_depth=3) ===")
dtree = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree.fit(X_train, y_train)
y_pred_tree = dtree.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree accuracy: {acc_tree:.4f}")
print("Decision Tree classification report:")
print(classification_report(y_test, y_pred_tree, target_names=iris.target_names))
print(f"Compared with KNN (unscaled) accuracy {acc_knn_unscaled:.4f}, the Decision Tree is lower by {acc_knn_unscaled - acc_tree:.4f}.")
# On this split, Decision Tree performs slightly worse than KNN.
# Scaling usually does not change Decision Tree results because tree splits are threshold-based, not distance-based.


# --- Logistic Regression and Regularization ---
# Q1
print("\n=== Logistic Regression Q1 (Regularization Strength via C) ===")
for c_value in [0.01, 1.0, 100.0]:
	base_lr = LogisticRegression(C=c_value, max_iter=1000, solver='liblinear')
	lr_model = OneVsRestClassifier(base_lr)
	lr_model.fit(X_train_scaled, y_train)
	coef_total_size = np.abs(np.vstack([est.coef_ for est in lr_model.estimators_])).sum()
	print(f"C={c_value:>6}: total |coefficients| sum = {coef_total_size:.6f}")
# As C increases (weaker regularization), total coefficient magnitude usually increases.
# This shows regularization shrinks coefficients toward zero to control model complexity.


# --- PCA ---
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# Q1
print("\n=== PCA Q1 ===")
print(f"X_digits shape: {X_digits.shape}")
print(f"images shape: {images.shape}")

fig_samples, axes_samples = plt.subplots(1, 10, figsize=(14, 2.5))
for digit in range(10):
	idx = np.where(y_digits == digit)[0][0]
	axes_samples[digit].imshow(images[idx], cmap='gray_r')
	axes_samples[digit].set_title(str(digit))
	axes_samples[digit].axis('off')
fig_samples.suptitle('One Example of Each Digit (0-9)')
fig_samples.tight_layout()
fig_samples.savefig('outputs/sample_digits.png', dpi=150)
plt.close(fig_samples)

# Q2
print("\n=== PCA Q2 ===")
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
scatter = ax_scatter.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)
fig_scatter.colorbar(scatter, ax=ax_scatter, label='Digit')
ax_scatter.set_title('Digits Projected onto First Two Principal Components')
ax_scatter.set_xlabel('PC1 score')
ax_scatter.set_ylabel('PC2 score')
fig_scatter.tight_layout()
fig_scatter.savefig('outputs/pca_2d_projection.png', dpi=150)
plt.close(fig_scatter)
# Same-digit images tend to form partial clusters in this 2D view, though many classes still overlap.

# Q3
print("\n=== PCA Q3 ===")
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_80 = int(np.argmax(cumulative_variance >= 0.80) + 1)
print(f"Components needed for ~80% explained variance: {n_components_80}")

fig_var, ax_var = plt.subplots(figsize=(8, 5))
ax_var.plot(
	np.arange(1, len(cumulative_variance) + 1),
	cumulative_variance,
	marker='o',
	markersize=3
)
ax_var.axhline(0.80, color='red', linestyle='--', label='80% variance')
ax_var.axvline(n_components_80, color='green', linestyle='--', label=f'{n_components_80} components')
ax_var.set_title('Cumulative Explained Variance by Number of PCA Components')
ax_var.set_xlabel('Number of components')
ax_var.set_ylabel('Cumulative explained variance ratio')
ax_var.legend()
fig_var.tight_layout()
fig_var.savefig('outputs/pca_variance_explained.png', dpi=150)
plt.close(fig_var)
# The curve suggests you need around this many components to capture 80% of total variance.


def reconstruct_digit(sample_idx, scores, pca, n_components):
	"""Reconstruct one digit using the first n_components principal components."""
	reconstruction = pca.mean_.copy()
	for i in range(n_components):
		reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
	return reconstruction.reshape(8, 8)


# Q4
print("\n=== PCA Q4 ===")
recon_components = [2, 5, 15, 40]
sample_indices = [0, 1, 2, 3, 4]

rows = 1 + len(recon_components)
cols = len(sample_indices)
fig_recon, axes_recon = plt.subplots(rows, cols, figsize=(10, 8))

# Original row
for col, idx in enumerate(sample_indices):
	axes_recon[0, col].imshow(images[idx], cmap='gray_r')
	axes_recon[0, col].axis('off')
	if col == 0:
		axes_recon[0, col].set_ylabel('Original', rotation=90, labelpad=12)

# Reconstruction rows
for row, n in enumerate(recon_components, start=1):
	for col, idx in enumerate(sample_indices):
		recon_img = reconstruct_digit(idx, scores, pca, n)
		axes_recon[row, col].imshow(recon_img, cmap='gray_r')
		axes_recon[row, col].axis('off')
		if col == 0:
			axes_recon[row, col].set_ylabel(f'n={n}', rotation=90, labelpad=18)

fig_recon.suptitle('PCA Reconstructions of First 5 Digits')
fig_recon.tight_layout()
fig_recon.savefig('outputs/pca_reconstructions.png', dpi=150)
plt.close(fig_recon)
print("Saved PCA reconstruction grid for n = 2, 5, 15, 40.")
# Digits become clearly recognizable around n=15 for most samples, which generally matches where the variance curve begins to level off.
