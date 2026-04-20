import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

# ── 0. Reproducibility ────────────────────────────────────────
np.random.seed(42)

# ── 1. Load Dataset ───────────────────────────────────────────
df = pd.read_csv("Cloud_Dataset.csv")

df["target"] = (
    (df["utilization"]  < 30) &
    (df["cpu_usage"]    < 50) &
    (df["memory_usage"] < 60)
).astype(int)

noise = np.random.rand(len(df)) < 0.1
df.loc[noise, "target"] = 1 - df.loc[noise, "target"]

# ── 2. Class distribution ─────────────────────────────────────
print("\n===== CLASS DISTRIBUTION =====")
counts = df["target"].value_counts()
print(f"Efficient   (0): {counts.get(0,0)}  ({counts.get(0,0)/len(df)*100:.1f}%)")
print(f"Inefficient (1): {counts.get(1,0)}  ({counts.get(1,0)/len(df)*100:.1f}%)")

# ── 3. Features ───────────────────────────────────────────────
features = ["utilization","cpu_usage","memory_usage",
            "disk_io","latency_ms","throughput","cost"]
X = df[features]
y = df["target"]

# ── 4. Train / Test split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")

# ── 5. Define all 4 models ────────────────────────────────────
#
#   Model 1 : Random Forest      — ensemble of decision trees
#   Model 2 : Gradient Boosting  — trees built sequentially, each fixing previous errors
#   Model 3 : Logistic Regression— simple linear probability model (needs scaling)
#   Model 4 : SVM                — finds optimal boundary between classes (needs scaling)
#
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42,
        learning_rate=0.1
    ),
    "Logistic Regression": Pipeline([          # Pipeline auto-scales before fitting
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        ))
    ]),
    "SVM": Pipeline([                          # SVM also needs scaling
        ("scaler", StandardScaler()),
        ("clf",    SVC(
            probability=True,                  # needed for ROC curve
            random_state=42,
            class_weight="balanced"
        ))
    ]),
}

# ── 6. Train, evaluate, collect results ──────────────────────
cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results  = {}
roc_data = {}

print("\n===== MODEL COMPARISON =====")
print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'AUC':>7} {'CV-F1':>8}")
print("-" * 75)

for name, clf in models.items():
    # Train
    clf.fit(X_train, y_train)

    # Predict
    pred      = clf.predict(X_test)
    pred_prob = clf.predict_proba(X_test)[:, 1]

    # Metrics
    acc  = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec  = recall_score(y_test, pred, zero_division=0)
    f1   = f1_score(y_test, pred, zero_division=0)
    cm   = confusion_matrix(y_test, pred)
    fpr, tpr, _ = roc_curve(y_test, pred_prob)
    roc_auc     = auc(fpr, tpr)

    # Cross-validation (fresh clone not needed — Pipeline/estimator is refit each fold)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1")

    results[name] = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "auc": roc_auc, "cv_f1": cv_scores.mean(),
        "cv_std": cv_scores.std(), "cm": cm
    }
    roc_data[name] = (fpr, tpr, roc_auc)

    print(f"{name:<22} {acc:>9.4f} {prec:>10.4f} {rec:>8.4f} {f1:>7.4f} {roc_auc:>7.4f} {cv_scores.mean():>8.4f}")

# ── 7. Best model ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["f1"])
best      = results[best_name]
print(f"\nBest model by F1: {best_name}  (F1 = {best['f1']:.4f})")

# Save best model
joblib.dump(models[best_name], "cloud_model_best.pkl")
print(f"Best model saved as cloud_model_best.pkl")

# ── 8. Save metrics ───────────────────────────────────────────
with open("model_comparison.txt", "w") as f:
    f.write("MODEL COMPARISON RESULTS\n========================\n\n")
    for name, r in results.items():
        f.write(f"{name}\n")
        f.write(f"  Accuracy  : {r['accuracy']:.4f}\n")
        f.write(f"  Precision : {r['precision']:.4f}\n")
        f.write(f"  Recall    : {r['recall']:.4f}\n")
        f.write(f"  F1 Score  : {r['f1']:.4f}\n")
        f.write(f"  ROC-AUC   : {r['auc']:.4f}\n")
        f.write(f"  CV F1     : {r['cv_f1']:.4f} (+/- {r['cv_std']:.4f})\n\n")
    f.write(f"Best model: {best_name}\n")
print("Comparison saved as model_comparison.txt")


# ══════════════════════════════════════════════════════════════
#   VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

DARK   = "#0f0f0f"
PANEL  = "#1a1a1a"
BORDER = "#2e2e2e"
WHITE  = "#f0f0f0"
MUTED  = "#888888"
MODEL_COLORS = {
    "Random Forest":      "#4f9cf9",
    "Gradient Boosting":  "#4fd97b",
    "Logistic Regression":"#f9c84f",
    "SVM":                "#f97b4f",
}
names = list(results.keys())

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.set_title(title, color=WHITE, fontsize=11, fontweight="bold", pad=10)

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── Plot 1: Grouped metric bar chart ──────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "1. All Metrics — All Models")

metrics     = ["accuracy", "precision", "recall", "f1", "auc"]
metric_lbls = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
x     = np.arange(len(metrics))
width = 0.18

for i, name in enumerate(names):
    vals = [results[name][m] for m in metrics]
    bars = ax1.bar(x + i * width, vals, width, label=name,
                   color=MODEL_COLORS[name], zorder=3)

ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(metric_lbls, color=MUTED, fontsize=8)
ax1.set_ylim(0, 1.18)
ax1.set_ylabel("Score")
ax1.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
ax1.set_axisbelow(True)
ax1.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE,
           fontsize=7, loc="upper right")

# ── Plot 2: F1 Score comparison (clean) ───────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "2. F1 Score Comparison")

f1_vals   = [results[n]["f1"] for n in names]
bar_cols  = [MODEL_COLORS[n] for n in names]
short     = ["RF", "GB", "LR", "SVM"]

bars2 = ax2.bar(short, f1_vals, color=bar_cols, width=0.5, zorder=3)
ax2.set_ylim(0, 1.15)
ax2.set_ylabel("F1 Score")
ax2.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
ax2.set_axisbelow(True)

for bar, val, name in zip(bars2, f1_vals, names):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{val:.3f}", ha="center", va="bottom", color=WHITE, fontsize=10, fontweight="bold")
    if name == best_name:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.07,
                 "BEST", ha="center", va="bottom", color="#4fd97b", fontsize=8, fontweight="bold")

# ── Plot 3: ROC Curves — all 4 on same axes ───────────────────
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, "3. ROC Curves — All Models")

ax3.plot([0,1],[0,1], color=MUTED, linestyle="--", linewidth=1, label="Random")
for name in names:
    fpr_, tpr_, rauc = roc_data[name]
    ax3.plot(fpr_, tpr_, color=MODEL_COLORS[name], linewidth=2,
             label=f"{name[:4]}.. AUC={rauc:.2f}")

ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=8)
ax3.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5)
ax3.set_axisbelow(True)

# ── Plot 4: Cross-validation F1 with error bars ───────────────
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4, "4. Cross-Validation F1 (mean ± std)")

cv_means = [results[n]["cv_f1"]  for n in names]
cv_stds  = [results[n]["cv_std"] for n in names]
cols4    = [MODEL_COLORS[n] for n in names]

bars4 = ax4.bar(short, cv_means, color=cols4, width=0.5,
                yerr=cv_stds, capsize=6, error_kw={"color": WHITE, "linewidth": 1.5},
                zorder=3)
ax4.set_ylim(0, 1.15)
ax4.set_ylabel("CV F1 Score")
ax4.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
ax4.set_axisbelow(True)

for bar, val in zip(bars4, cv_means):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
             f"{val:.3f}", ha="center", va="bottom", color=WHITE, fontsize=9)

# ── Plot 5: Confusion matrices — 2x2 grid of heatmaps ─────────
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5, "5. Recall vs Precision Tradeoff")

rec_vals  = [results[n]["recall"]    for n in names]
prec_vals = [results[n]["precision"] for n in names]

for i, name in enumerate(names):
    ax5.scatter(rec_vals[i], prec_vals[i],
                color=MODEL_COLORS[name], s=180, zorder=5, label=name)
    ax5.annotate(short[i],
                 (rec_vals[i], prec_vals[i]),
                 textcoords="offset points", xytext=(8, 4),
                 color=MODEL_COLORS[name], fontsize=9, fontweight="bold")

ax5.set_xlabel("Recall  (catching wasteful resources)")
ax5.set_ylabel("Precision  (alert accuracy)")
ax5.set_xlim(0, 1.1); ax5.set_ylim(0, 1.1)
ax5.yaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5)
ax5.xaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5)
ax5.set_axisbelow(True)
ax5.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=8)

# ideal corner marker
ax5.scatter([1],[1], marker="*", s=200, color="#f9c84f", zorder=6)
ax5.annotate("ideal", (1,1), textcoords="offset points",
             xytext=(-28,4), color="#f9c84f", fontsize=8)

# ── Plot 6: Summary heatmap table ─────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6, "6. Metric Heatmap (all models)")

table_metrics = ["accuracy","precision","recall","f1","auc","cv_f1"]
table_lbls    = ["Accuracy","Precision","Recall","F1","AUC","CV-F1"]
data_matrix   = np.array([[results[n][m] for m in table_metrics] for n in names])

im = ax6.imshow(data_matrix, cmap="YlGn", vmin=0.4, vmax=1.0, aspect="auto")
plt.colorbar(im, ax=ax6).ax.tick_params(colors=MUTED)

ax6.set_xticks(range(len(table_lbls)))
ax6.set_xticklabels(table_lbls, color=MUTED, fontsize=8, rotation=30, ha="right")
ax6.set_yticks(range(len(names)))
ax6.set_yticklabels(["RF","GB","LR","SVM"], color=MUTED, fontsize=9)

for i in range(len(names)):
    for j in range(len(table_metrics)):
        ax6.text(j, i, f"{data_matrix[i,j]:.2f}",
                 ha="center", va="center", fontsize=9, fontweight="bold",
                 color=DARK if data_matrix[i,j] > 0.65 else WHITE)

# ── Main title ────────────────────────────────────────────────
fig.suptitle(
    f"Cloud Resource Efficiency — Model Comparison  |  Best: {best_name} (F1={best['f1']:.3f})",
    color=WHITE, fontsize=14, fontweight="bold", y=0.99
)

plt.savefig("model_comparison_report.png", dpi=150,
            bbox_inches="tight", facecolor=DARK)
plt.show()
print("\nComparison chart saved as model_comparison_report.png")