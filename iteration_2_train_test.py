import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("credit_default.csv")

# Separate features and target
X = df.drop("default payment next month", axis=1)
y = df["default payment next month"]

# Encode categorical variables if present
X = pd.get_dummies(X, drop_first=True)

# Split into 80% training and 20% test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,   # 20% test
    random_state=42,
    stratify=y        # optional but recommended for classification
)

# Train decision tree on training set only
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate only on test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.4f}")
