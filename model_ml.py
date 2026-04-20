import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.utils.class_weight import compute_class_weight

np.random.seed(42)

df = pd.read_csv("Cloud_Dataset.csv")

df["target"] = (
    (df["utilization"] < 30) &
    (df["cpu_usage"]   < 50) &
    (df["memory_usage"]< 60)
).astype(int)

noise = np.random.rand(len(df)) < 0.1
df.loc[noise, "target"] = 1 - df.loc[noise, "target"]

print("\n===== CLASS DISTRIBUTION =====")
counts = df["target"].value_counts()
print(f"Efficient   (0): {counts.get(0, 0)}  ({counts.get(0,0)/len(df)*100:.1f}%)")
print(f"Inefficient (1): {counts.get(1, 0)}  ({counts.get(1,0)/len(df)*100:.1f}%)")

imbalance_ratio = counts.get(0, 1) / counts.get(1, 1)
if imbalance_ratio > 2:
    print(f"WARNING: Class imbalance detected (ratio {imbalance_ratio:.1f}:1) — using class_weight='balanced'")
else:
    print("Class distribution looks balanced.")

features = [
    "utilization",
    "cpu_usage",
    "memory_usage",
    "disk_io",
    "latency_ms",
    "throughput",
    "cost"
]

X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDataset split — Train: {len(X_train)} rows | Test: {len(X_test)} rows")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

pred      = model.predict(X_test)
pred_prob = model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall    = recall_score(y_test, pred)
f1        = f1_score(y_test, pred)
cm        = confusion_matrix(y_test, pred)
fpr, tpr, _ = roc_curve(y_test, pred_prob)
roc_auc     = auc(fpr, tpr)

fresh_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(fresh_model, X, y, cv=cv, scoring="f1")

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy        : {accuracy:.4f}")
print(f"Precision       : {precision:.4f}")
print(f"Recall          : {recall:.4f}")
print(f"F1 Score        : {f1:.4f}")
print(f"ROC-AUC         : {roc_auc:.4f}")
print(f"Cross-val F1    : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"Confusion Matrix:\n{cm}")

joblib.dump(model, "cloud_model.pkl")
print("\nModel saved as cloud_model.pkl")

with open("model_metrics.txt", "w") as f:
    f.write("MODEL PERFORMANCE\n=================\n")
    f.write(f"Accuracy          : {accuracy:.4f}\n")
    f.write(f"Precision         : {precision:.4f}\n")
    f.write(f"Recall            : {recall:.4f}\n")
    f.write(f"F1 Score          : {f1:.4f}\n")
    f.write(f"ROC-AUC           : {roc_auc:.4f}\n")
    f.write(f"Cross-val F1 Mean : {cv_scores.mean():.4f}\n")
    f.write(f"Cross-val F1 Std  : {cv_scores.std():.4f}\n")
    f.write(f"Confusion Matrix  :\n{cm}\n")
    f.write(f"\nClass Distribution:\n{counts.to_string()}\n")

print("Metrics saved as model_metrics.txt")


# ══════════════════════════════════════════════════════════════
#   5 METRIC VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#0f0f0f")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

DARK   = "#0f0f0f"
PANEL  = "#1a1a1a"
BORDER = "#2e2e2e"
WHITE  = "#f0f0f0"
MUTED  = "#888888"
COLORS = ["#4f9cf9", "#f97b4f", "#4fd97b", "#f9c84f", "#c44ff9", "#f94f6e", "#4ff9f0"]
BLUE   = "#4f9cf9"
GREEN  = "#4fd97b"
RED    = "#f94f6e"

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.set_title(title, color=WHITE, fontsize=11, fontweight="bold", pad=10)

# ── Plot 1: Metrics Bar Chart ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "1. Model Performance Metrics")

metric_names  = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
metric_values = [accuracy, precision, recall, f1, roc_auc]

bars = ax1.bar(metric_names, metric_values, color=COLORS[:5], width=0.55, zorder=3)
ax1.set_ylim(0, 1.15)
ax1.set_ylabel("Score")
ax1.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.6, zorder=0)
ax1.set_axisbelow(True)

for bar, val in zip(bars, metric_values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{val:.2f}", ha="center", va="bottom", color=WHITE, fontsize=9, fontweight="bold")

ax1.tick_params(axis="x", rotation=15)

# ── Plot 2: Confusion Matrix Heatmap ──────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "2. Confusion Matrix")

im = ax2.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax2).ax.tick_params(colors=MUTED)

classes = ["Efficient (0)", "Inefficient (1)"]
ax2.set_xticks([0, 1]); ax2.set_xticklabels(classes, color=MUTED, fontsize=8)
ax2.set_yticks([0, 1]); ax2.set_yticklabels(classes, color=MUTED, fontsize=8, rotation=90, va="center")
ax2.set_xlabel("Predicted Label")
ax2.set_ylabel("True Label")

thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold",
                 color=WHITE if cm[i, j] > thresh else PANEL)

# ── Plot 3: ROC Curve ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, "3. ROC Curve")

ax3.plot(fpr, tpr, color=BLUE, linewidth=2, label=f"AUC = {roc_auc:.2f}")
ax3.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Random")
ax3.fill_between(fpr, tpr, alpha=0.08, color=BLUE)
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=9)
ax3.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.6)
ax3.set_axisbelow(True)

# ── Plot 4: Cross-Validation F1 Scores ────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4, "4. Cross-Validation F1 Scores (5-Fold)")

fold_labels = [f"Fold {i+1}" for i in range(len(cv_scores))]
bar_colors  = [GREEN if s >= cv_scores.mean() else RED for s in cv_scores]

bars4 = ax4.bar(fold_labels, cv_scores, color=bar_colors, width=0.5, zorder=3)
ax4.axhline(cv_scores.mean(), color=BLUE, linestyle="--", linewidth=1.5,
            label=f"Mean = {cv_scores.mean():.2f}")
ax4.set_ylim(max(0, cv_scores.min() - 0.05), min(1.1, cv_scores.max() + 0.08))
ax4.set_ylabel("F1 Score")
ax4.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.6, zorder=0)
ax4.set_axisbelow(True)
ax4.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=9)

for bar, val in zip(bars4, cv_scores):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{val:.3f}", ha="center", va="bottom", color=WHITE, fontsize=8)

# ── Plot 5: Feature Importance (now includes utilization) ──────
ax5 = fig.add_subplot(gs[1, 1:])
style_ax(ax5, "5. Feature Importance (7 features)")

importances = model.feature_importances_
sorted_idx  = np.argsort(importances)
sorted_feat = [features[i] for i in sorted_idx]
sorted_imp  = importances[sorted_idx]
bar_clrs    = [COLORS[i % len(COLORS)] for i in range(len(sorted_feat))]

bars5 = ax5.barh(sorted_feat, sorted_imp, color=bar_clrs, height=0.55, zorder=3)
ax5.set_xlabel("Importance Score")
ax5.xaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.6, zorder=0)
ax5.set_axisbelow(True)
ax5.tick_params(axis="y", labelsize=10, colors=WHITE)

for bar, val in zip(bars5, sorted_imp):
    ax5.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", color=WHITE, fontsize=9)

# ── Main Title ─────────────────────────────────────────────────
fig.suptitle(
    "Cloud Resource Efficiency — ML Model Evaluation",
    color=WHITE, fontsize=15, fontweight="bold", y=0.98
)

plt.savefig("model_metrics_report.png", dpi=150,
            bbox_inches="tight", facecolor=DARK)
plt.show()
print("\nChart saved as model_metrics_report.png")