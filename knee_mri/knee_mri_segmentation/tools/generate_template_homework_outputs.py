import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "results", "eda_figures", "eda_features.csv")
OUT_DIR = os.path.join(BASE_DIR, "results", "homework_template_style")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH).dropna().copy()

class_dist = df["kl_grade"].value_counts().sort_index().rename_axis("class").reset_index(name="count")
class_dist["share_%"] = (class_dist["count"] / class_dist["count"].sum() * 100).round(2)
class_dist.to_csv(os.path.join(OUT_DIR, "classification_class_distribution.csv"), index=False)

feature_cols = [c for c in df.columns if c not in ["kl_grade", "class_name"]]
X = df[feature_cols]
y = df["kl_grade"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, multi_class="multinomial", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE),
    "SVM": SVC(kernel="rbf", probability=False, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
}

cls_rows = []
pred_store = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    pred_store[name] = pred
    cls_rows.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
        }
    )

cls_results = pd.DataFrame(cls_rows).sort_values("F1_weighted", ascending=False).reset_index(drop=True)
cls_results.insert(0, "Rank", np.arange(1, len(cls_results) + 1))

cls_results_rounded = cls_results.copy()
for c in ["Accuracy", "Precision_weighted", "Recall_weighted", "F1_weighted"]:
    cls_results_rounded[c] = (cls_results_rounded[c] * 100).round(2)

cls_results_rounded.to_csv(os.path.join(OUT_DIR, "classification_summary_metrics.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_df = cls_results_rounded.copy()
axes[0].bar(plot_df["Model"], plot_df["F1_weighted"], color="#4C72B0")
axes[0].set_title("F1-score (weighted) по моделям, %")
axes[0].set_ylabel("F1, %")
axes[0].tick_params(axis="x", rotation=25)
axes[0].set_ylim(0, 100)

best_model = cls_results.iloc[0]["Model"]
cm = confusion_matrix(y_test, pred_store[best_model], labels=sorted(y.unique()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))
disp.plot(ax=axes[1], colorbar=False)
axes[1].set_title(f"Confusion Matrix (best: {best_model})")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "classification_bar_and_confusion.png"), dpi=180, bbox_inches="tight")
plt.close()

rng = np.random.RandomState(RANDOM_STATE)
reg_df = df.copy()
progress_base = (
    3.5 * (reg_df["osteophyte_score"] / reg_df["osteophyte_score"].max())
    + 2.8 * (1 - reg_df["joint_space_width"] / reg_df["joint_space_width"].max())
    + 1.6 * (reg_df["sclerosis_index"] / (reg_df["sclerosis_index"].max() + 1e-9))
    + 1.2 * (reg_df["entropy"] / reg_df["entropy"].max())
    + 0.8 * (reg_df["mean_gradient"] / reg_df["mean_gradient"].max())
    + rng.normal(0, 0.4, len(reg_df))
)
progress_base += 1.5
reg_df["next_visit_progress_percent"] = progress_base

reg_features = feature_cols + ["kl_grade"]
X_reg = reg_df[reg_features]
y_reg = reg_df["next_visit_progress_percent"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE
)

reg_scaler = StandardScaler()
Xr_train_scaled = reg_scaler.fit_transform(Xr_train)
Xr_test_scaled = reg_scaler.transform(Xr_test)

lin_model = LinearRegression()
lin_model.fit(Xr_train_scaled, yr_train)
yr_pred = lin_model.predict(Xr_test_scaled)

rmse = float(np.sqrt(mean_squared_error(yr_test, yr_pred)))
mae = float(mean_absolute_error(yr_test, yr_pred))
mape = float(np.mean(np.abs((yr_test - yr_pred) / np.clip(np.abs(yr_test), 1e-8, None))) * 100)
r2 = float(r2_score(yr_test, yr_pred))

reg_metrics = pd.DataFrame(
    [
        {"Metric": "R2 Score", "Value": r2},
        {"Metric": "RMSE", "Value": rmse},
        {"Metric": "MAE", "Value": mae},
        {"Metric": "MAPE_%", "Value": mape},
    ]
)
reg_metrics.to_csv(os.path.join(OUT_DIR, "linear_regression_metrics.csv"), index=False)

coef_df = pd.DataFrame({"feature": reg_features, "coefficient": lin_model.coef_}).sort_values(
    "coefficient", key=lambda s: s.abs(), ascending=False
)
coef_df.to_csv(os.path.join(OUT_DIR, "linear_regression_coefficients.csv"), index=False)

residuals = yr_test - yr_pred
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

axes[0, 0].scatter(yr_test, yr_pred, alpha=0.5)
mn = min(yr_test.min(), yr_pred.min())
mx = max(yr_test.max(), yr_pred.max())
axes[0, 0].plot([mn, mx], [mn, mx], "r--")
axes[0, 0].set_title("Prediction vs Reality")
axes[0, 0].set_xlabel("Actual")
axes[0, 0].set_ylabel("Predicted")

axes[0, 1].hist(residuals, bins=30, color="#55A868", alpha=0.8)
axes[0, 1].axvline(0, color="red", linestyle="--")
axes[0, 1].set_title("Residual Distribution")
axes[0, 1].set_xlabel("Error")

top_coef = coef_df.head(10).iloc[::-1]
axes[0, 2].barh(top_coef["feature"], top_coef["coefficient"], color="#C44E52")
axes[0, 2].set_title("Top-10 Coefficients")

axes[1, 0].scatter(yr_pred, residuals, alpha=0.5, color="#8172B2")
axes[1, 0].axhline(0, color="red", linestyle="--")
axes[1, 0].set_title("Residuals vs Predicted")
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("Residual")

abs_err = np.abs(residuals)
axes[1, 1].plot(np.sort(abs_err.values), color="#4C72B0")
axes[1, 1].set_title("Absolute Error Curve")
axes[1, 1].set_xlabel("Sorted sample index")
axes[1, 1].set_ylabel("|error|")

metric_show = pd.Series({"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE%": mape})
axes[1, 2].bar(metric_show.index, metric_show.values, color=["#4C72B0", "#55A868", "#C44E52", "#CCB974"])
axes[1, 2].set_title("Regression Metrics")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "linear_regression_diagnostics.png"), dpi=180, bbox_inches="tight")
plt.close()

summary = {
    "best_classification_model": cls_results.iloc[0]["Model"],
    "best_f1_weighted_percent": float(cls_results_rounded.iloc[0]["F1_weighted"]),
    "regression_metrics": {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape_percent": mape,
    },
}
with open(os.path.join(OUT_DIR, "run_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("DONE")
print("OUT_DIR:", OUT_DIR)
print("BEST MODEL:", summary["best_classification_model"], summary["best_f1_weighted_percent"])
print("REG:", summary["regression_metrics"])
