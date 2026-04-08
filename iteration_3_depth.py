import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support

# Load dataset
df = pd.read_csv("credit_default.csv")

# Features and target
X = df.drop('default.payment.next.month', axis=1)
y = df['default.payment.next.month']

# Convert categorical columns if any
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

depths = list(range(2, 16))
precisions = []
recalls = []
f1_scores = []

best_depth = None
best_f1 = -1

for depth in depths:
    # Train model with current depth
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Get precision, recall, F1 for class 1 only
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[1], average=None, zero_division=0
    )

    p = precision[0]
    r = recall[0]
    f = f1[0]

    precisions.append(p)
    recalls.append(r)
    f1_scores.append(f)

    # Track best depth by F1 score
    if f > best_f1:
        best_f1 = f
        best_depth = depth

# Print metrics for each depth
print("Depth | Precision | Recall | F1 Score")
print("--------------------------------------")
for d, p, r, f in zip(depths, precisions, recalls, f1_scores):
    print(f"{d:5d} | {p:.4f}    | {r:.4f} | {f:.4f}")

print(f"\nBest depth based on F1 score: {best_depth}")
print(f"Best F1 score: {best_f1:.4f}")

# Plot metrics vs tree depth
plt.figure(figsize=(10, 6))
plt.plot(depths, precisions, marker='o', label='Precision')
plt.plot(depths, recalls, marker='o', label='Recall')
plt.plot(depths, f1_scores, marker='o', label='F1 Score')

plt.xlabel("Tree Depth")
plt.ylabel("Score")
plt.title("Precision, Recall, and F1 Score vs Decision Tree Depth")
plt.xticks(depths)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
