import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("credit_default.csv")

# Drop ID column if it exists
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

# Define features and target
target_col = "default.payment.next.month"
X = df.drop(columns=[target_col])
y = df[target_col]

# get_dummies not needed - all features are already numeric
# X = pd.get_dummies(X, drop_first=True)

# -----------------------------
# Train/test split (80/20)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train decision tree
# -----------------------------
model = DecisionTreeClassifier(
    max_depth=3,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------
# Predictions and evaluation
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Default", "Default"],
        zero_division=0
    )
)

# -----------------------------
# Confusion matrix heatmap
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Default", "Default"],
    yticklabels=["No Default", "Default"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

print("Saved confusion matrix heatmap to: confusion_matrix.png")

# -----------------------------
# Feature importance chart
# -----------------------------
feature_importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

top_10 = feature_importances.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_10.values, y=top_10.index)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance_top10.png", dpi=300)
plt.close()

print("Saved feature importance chart to: feature_importance_top10.png")
